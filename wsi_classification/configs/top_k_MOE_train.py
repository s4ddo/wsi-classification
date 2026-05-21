"""Train SlideMoE on Virchow2 features.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --override train.lr=2e-4 model.num_experts=8

A split JSON must already exist (see splits/make_splits.py).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from data import SlideH5Dataset, pad_collate
from model import SlideMoE


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str, overrides: Optional[List[str]] = None) -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        for kv in overrides:
            if "=" not in kv:
                raise ValueError(f"Override must be key=value, got {kv}")
            key, val = kv.split("=", 1)
            try:
                parsed: Any = yaml.safe_load(val)
            except yaml.YAMLError:
                parsed = val
            d = cfg
            parts = key.split(".")
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = parsed
    return cfg


def cosine_warmup_lr(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
    y_pred = (y_score > 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y_true, y_score)) if len(set(y_true)) > 1 else float("nan"),
        "acc": float(accuracy_score(y_true, y_pred)),
    }


def build_loader(cfg: Dict[str, Any], split: str, shuffle: bool) -> DataLoader:
    ds_cfg = cfg["data"]
    ds = SlideH5Dataset(
        h5_dir=ds_cfg["h5_dir"],
        split_json=ds_cfg["split_json"],
        split=split,
        reference_csv=ds_cfg.get("reference_csv"),
        feature_key=ds_cfg["feature_key"],
        coord_key=ds_cfg["coord_key"],
        label_key=ds_cfg["label_key"],
        max_patches=ds_cfg.get("max_patches"),
        seed=cfg["train"]["seed"] + (0 if split == "train" else 1),
    )
    return DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"] if split == "train" else 1,
        shuffle=shuffle,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=pad_collate,
        pin_memory=True,
        persistent_workers=cfg["train"]["num_workers"] > 0,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_y, all_s = [], []
    loss_sum, n = 0.0, 0
    bce_eval = nn.BCEWithLogitsLoss(reduction="sum")
    for batch in loader:
        feats = batch["features"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True).float()
        out = model(feats, mask)
        logits = out["logits"].squeeze(-1)
        loss_sum += bce_eval(logits, labels).item()
        n += labels.numel()
        prob = torch.sigmoid(logits)
        all_y.append(labels.detach().cpu().numpy())
        all_s.append(prob.detach().cpu().numpy())
    y = np.concatenate(all_y)
    s = np.concatenate(all_s)
    metrics = compute_metrics(y, s)
    # Plain classification BCE (no aux / no load-balance) for table comparability.
    metrics["loss"] = loss_sum / max(n, 1)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--no_wandb", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config, args.override)
    set_seed(cfg["train"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    script_t0 = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    ckpt_dir = Path(cfg["logging"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = cfg["logging"].get("wandb", False) and not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=cfg["logging"]["wandb_project"],
                entity=cfg["logging"].get("wandb_entity"),
                name=cfg["logging"].get("run_name"),
                config=cfg,
            )
        except Exception as e:  # no API key / offline cluster — keep training
            print(f"[wandb] disabled ({type(e).__name__}: {e}); logging to stdout only")
            use_wandb = False

    train_loader = build_loader(cfg, "train", shuffle=True)
    val_loader = build_loader(cfg, "val", shuffle=False)

    model = SlideMoE(
        in_dim=cfg["data"]["feature_dim"],
        **{k: v for k, v in cfg["model"].items()},
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params_total = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params/1e6:.2f}M trainable / {n_params_total/1e6:.2f}M total")

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"]["amp"] and device.type == "cuda") if device.type == "cuda" else torch.amp.GradScaler(device.type, enabled=False)

    pos_weight = cfg["train"].get("pos_class_weight")
    pos_weight_t = (
        torch.tensor([float(pos_weight)], device=device) if pos_weight is not None else None
    )
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)

    accum = max(1, int(cfg["train"].get("grad_accum_steps", 1)))
    # LR schedule operates in OPTIMIZER-step units (same units as `step`), not
    # micro-batch units — otherwise the cosine barely decays over the run.
    n_batches = len(train_loader)
    steps_per_epoch = math.ceil(n_batches / accum)
    max_steps = cfg["train"].get("max_steps")  # gradient-step budget (None = use epochs)
    total_steps = steps_per_epoch * cfg["train"]["num_epochs"]
    if max_steps is not None:
        total_steps = min(total_steps, int(max_steps))
    warmup_steps = min(
        steps_per_epoch * cfg["train"]["warmup_epochs"], max(1, total_steps - 1)
    )
    step = 0
    best_auroc = -1.0
    best_metrics: Dict[str, float] = {}

    for epoch in range(cfg["train"]["num_epochs"]):
        model.train()
        t0 = time.time()
        run_loss = run_cls = run_aux = run_lb = 0.0
        n_seen = 0
        optim.zero_grad(set_to_none=True)
        for bi, batch in enumerate(train_loader):
            feats = batch["features"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True).float()

            for g in optim.param_groups:
                g["lr"] = cosine_warmup_lr(step, total_steps, warmup_steps, cfg["train"]["lr"])

            with torch.cuda.amp.autocast(enabled=cfg["train"]["amp"] and device.type == "cuda"):
                out = model(feats, mask)
                logits = out["logits"].squeeze(-1)
                loss_cls = bce(logits, labels)
                loss_aux = (
                    bce(out["aux_logits"].squeeze(-1), labels)
                    if out["aux_logits"] is not None
                    else torch.zeros((), device=device)
                )
                loss_lb = out["load_balance_loss"]
                loss_full = (
                    loss_cls
                    + cfg["train"]["aux_scorer_coef"] * loss_aux
                    + cfg["train"]["load_balance_coef"] * loss_lb
                )
                loss = loss_full / accum

            scaler.scale(loss).backward()

            do_step = ((bi + 1) % accum == 0) or (bi + 1 == n_batches)
            if do_step:
                if cfg["train"]["grad_clip"] is not None:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                step += 1

            bs = feats.size(0)
            run_loss += loss_full.item() * bs
            run_cls += loss_cls.item() * bs
            run_aux += float(loss_aux.item()) * bs
            run_lb += float(loss_lb.item()) * bs
            n_seen += bs

            if use_wandb and do_step and step % cfg["logging"]["log_every"] == 0:
                wandb.log(
                    {
                        "train/loss": loss_full.item(),
                        "train/cls": loss_cls.item(),
                        "train/aux": float(loss_aux.item()),
                        "train/lb": float(loss_lb.item()),
                        "train/lr": optim.param_groups[0]["lr"],
                        "step": step,
                    }
                )

            if max_steps is not None and step >= max_steps:
                break

        val_metrics = evaluate(model, val_loader, device)
        epoch_dt = time.time() - t0
        print(
            f"[epoch {epoch:03d}] "
            f"loss={run_loss/n_seen:.4f} cls={run_cls/n_seen:.4f} "
            f"aux={run_aux/n_seen:.4f} lb={run_lb/n_seen:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_auroc={val_metrics['auroc']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"({epoch_dt:.1f}s)"
        )
        if use_wandb:
            wandb.log(
                {f"val/{k}": v for k, v in val_metrics.items()} | {"epoch": epoch}
            )

        # Save last checkpoint
        last_ckpt_path = ckpt_dir / "last.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "val_metrics": val_metrics,
            },
            last_ckpt_path,
        )
        
        # Save best checkpoint
        if val_metrics["auroc"] == val_metrics["auroc"] and val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            best_metrics = dict(val_metrics)
            best_metrics["epoch"] = epoch
            ckpt_path = ckpt_dir / "best.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                ckpt_path,
            )
            print(f"  -> saved {ckpt_path} (val_auroc={best_auroc:.4f})")
        else:
            print(f"  -> saved {last_ckpt_path} (last epoch)")

        if max_steps is not None and step >= max_steps:
            print(f"Reached max_steps={max_steps}; stopping at step {step}.")
            break

    total_seconds = time.time() - script_t0
    if device.type == "cuda":
        peak_alloc_gb = torch.cuda.max_memory_allocated(device) / 1024**3
        peak_reserved_gb = torch.cuda.max_memory_reserved(device) / 1024**3
    else:
        peak_alloc_gb = peak_reserved_gb = 0.0

    # Metrics at the best (selected) epoch — these are the table-comparable
    # numbers (the baseline table reports Val. Loss and Val. Acc., not AUROC).
    bm = best_metrics
    print(
        f"Best epoch {bm.get('epoch', -1)}: "
        f"val_loss={bm.get('loss', float('nan')):.4f} "
        f"val_acc={bm.get('acc', float('nan')):.4f} "
        f"val_auroc={bm.get('auroc', float('nan')):.4f}"
    )
    print(f"Params: {n_params/1e6:.2f}M trainable / {n_params_total/1e6:.2f}M total")
    print(f"Total runtime: {total_seconds:.1f} s ({total_seconds/60:.1f} min)")
    print(
        f"Peak VRAM: {peak_alloc_gb:.2f} GB allocated / "
        f"{peak_reserved_gb:.2f} GB reserved"
    )

    run_stats = {
        "best_epoch": bm.get("epoch"),
        "best_val_auroc": best_auroc,
        "best_val_acc": bm.get("acc"),
        "best_val_loss": bm.get("loss"),
        "best_val_auprc": bm.get("auprc"),
        "params_trainable": int(n_params),
        "params_total": int(n_params_total),
        "total_seconds": total_seconds,
        "peak_vram_alloc_gb": peak_alloc_gb,
        "peak_vram_reserved_gb": peak_reserved_gb,
        "num_epochs": cfg["train"]["num_epochs"],
        "grad_steps_total": step,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }
    with open(ckpt_dir / "run_stats.json", "w") as f:
        json.dump(run_stats, f, indent=2)
    print(f"Wrote {ckpt_dir / 'run_stats.json'}")

    if use_wandb:
        wandb.summary["best_val_auroc"] = best_auroc
        wandb.summary["total_seconds"] = total_seconds
        wandb.summary["peak_vram_alloc_gb"] = peak_alloc_gb
        wandb.finish()


if __name__ == "__main__":
    main()
