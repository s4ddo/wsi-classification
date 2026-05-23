"""VRAM benchmarking script for MIL models.

Tests multiple patch counts (N) and logs VRAM usage vs time to wandb.

Usage:
    python benchmark_vram.py --config configs/final_camely_abmil.py
    python benchmark_vram.py --config configs/final_camely_abmil.py --patch-counts 10000 50000 100000
"""

import argparse
import time
import torch
from pathlib import Path
import wandb

from wsi_classification.experiments.default_cfg import ExperimentConfig
from wsi_classification.experiments.utils.cli import load_config_from_file, apply_config_overrides
from wsi_classification.experiments.utils.lazy_config import instantiate
from wsi_classification.experiments.datamodules.h5_datamodule import H5FeatureBagDataModule


def parse_args():
    parser = argparse.ArgumentParser(description="VRAM Benchmark for MIL Models")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--patch-counts",
        nargs="+",
        type=int,
        default=[100000, 200000, 300000, 400000, 500000],
        help="List of patch counts to benchmark (default: 10000 30000 50000 70000 100000)",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=1280,
        help="Feature dimension (default: 1280)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=2,
        help="Number of warmup steps before measuring (default: 2)",
    )
    parser.add_argument(
        "--measure-steps",
        type=int,
        default=5,
        help="Number of steps to measure for averaging (default: 5)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="vram_benchmark",
        help="Wandb project name",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional config overrides",
    )
    return parser.parse_args()


def check_oom_before_benchmark(num_patches: int, feature_dim: int) -> bool:
    """Check if benchmark would OOM based on available GPU memory."""
    if not torch.cuda.is_available():
        return False

    # Get available GPU memory
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    free_memory = torch.cuda.mem_get_info()[0]  # Free memory in bytes
    total_memory = torch.cuda.mem_get_info()[1]

    # Conservative estimate: features + gradients + optimizer states + overhead
    # features: N * D * 4 bytes (float32)
    # gradients: ~same as features
    # optimizer states (Adam): 2x features for momentum buffers
    # activation overhead: ~20%
    feature_bytes = num_patches * feature_dim * 4
    estimated_needed = feature_bytes * 4  # *4 for features + grads + optimizer + overhead

    print(f"  Memory check: {estimated_needed/1024**3:.1f}GB needed, {free_memory/1024**3:.1f}GB free")

    # Leave 10% buffer
    if estimated_needed > free_memory * 0.9:
        return True  # Would OOM

    return False  # Should fit


def benchmark_n(
    config: ExperimentConfig,
    num_patches: int,
    feature_dim: int,
    warmup_steps: int,
    measure_steps: int,
) -> dict:
    """Benchmark a single N value.

    Returns dict with peak_vram_mb, avg_time_ms, num_patches.
    Raises RuntimeError with OOM message if out of memory.
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking N={num_patches:,} patches, D={feature_dim}")
    print(f"{'='*60}")

    # Pre-check for OOM
    if check_oom_before_benchmark(num_patches, feature_dim):
        print(f"OOM pre-check failed for N={num_patches}")
        raise RuntimeError("CUDA out of memory")

    # Create synthetic datamodule
    datamodule = H5FeatureBagDataModule(
        train_csv="dummy.csv",  # Required but not used for synthetic
        val_csv="dummy.csv",
        use_synthetic=True,
        synthetic_num_patches=num_patches,
        synthetic_feature_dim=feature_dim,
        batch_size=1,
        num_workers=0,
    )
    datamodule.prepare_data()
    datamodule.setup("fit")

    # Update config for this run
    config_copy = config
    config_copy.train.iterations = warmup_steps + measure_steps
    config_copy.train.do = True
    config_copy.test.do = False

    # Build model
    network = instantiate(config_copy.net, in_features=feature_dim, out_features=1)
    if config_copy.compile:
        network = torch.compile(network)
    model = instantiate(config_copy.lightning_wrapper_class, network=network, cfg=config_copy)

    # Create trainer with no logging to avoid clutter
    config_copy.wandb.project = None  # Disable wandb logging for individual runs

    # Setup for measurement
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, VRAM measurement will be 0")

    # Manual training loop for precise measurement
    model = model.to(device)
    model.train()

    optimizer = instantiate(config_copy.optimizer, params=model.parameters())

    train_loader = datamodule.train_dataloader()
    data_iter = iter(train_loader)

    # Warmup
    print(f"Warming up for {warmup_steps} steps...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for _ in range(warmup_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        optimizer.zero_grad()
        loss = model.training_step(batch, 0)
        loss.backward()
        optimizer.step()

    # Synchronize before measurement
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Measure
    print(f"Measuring for {measure_steps} steps...")
    times = []

    for step in range(measure_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        start = time.perf_counter()
        optimizer.zero_grad()
        loss = model.training_step(batch, step)
        loss.backward()
        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()

        times.append((end - start) * 1000)  # Convert to ms

    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    avg_time_ms = sum(times) / len(times)

    # Cleanup
    del model, network, optimizer, datamodule
    torch.cuda.empty_cache()

    results = {
        "num_patches": num_patches,
        "feature_dim": feature_dim,
        "peak_vram_mb": peak_vram_mb,
        "avg_time_ms": avg_time_ms,
        "min_time_ms": min(times),
        "max_time_ms": max(times),
    }

    print(f"Results: VRAM={peak_vram_mb:.1f} MB, Time={avg_time_ms:.2f} ms")
    return results


def main():
    args = parse_args()

    # Load base config
    config = load_config_from_file(args.config)
    if args.overrides:
        config = apply_config_overrides(config, args.overrides)

    # Get model name from config path
    model_name = Path(args.config).stem.replace("final_", "")

    # Initialize wandb for the benchmark
    wandb.init(
        project=args.project,
        name=f"{model_name}_benchmark",
        config={
            "model": model_name,
            "config_path": args.config,
            "patch_counts": args.patch_counts,
            "feature_dim": args.feature_dim,
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
        },
    )

    # Run benchmarks
    all_results = []
    for num_patches in args.patch_counts:
        try:
            result = benchmark_n(
                config=config,
                num_patches=num_patches,
                feature_dim=args.feature_dim,
                warmup_steps=args.warmup_steps,
                measure_steps=args.measure_steps,
            )
            all_results.append(result)

            # Log individual point
            wandb.log({
                "num_patches": num_patches,
                "peak_vram_mb": result["peak_vram_mb"],
                "avg_time_ms": result["avg_time_ms"],
            })

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM at N={num_patches}")
                wandb.log({
                    "num_patches": num_patches,
                    "oom": True,
                })
                all_results.append({
                    "num_patches": num_patches,
                    "oom": True,
                })
                torch.cuda.empty_cache()
            else:
                raise

    # Create summary table
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"{'N':>10} {'VRAM (MB)':>12} {'Time (ms)':>12} {'Status':>10}")
    print("-" * 60)
    for r in all_results:
        n = r["num_patches"]
        if r.get("oom"):
            print(f"{n:>10,} {'OOM':>12} {'OOM':>12} {'OOM':>10}")
        else:
            print(f"{n:>10,} {r['peak_vram_mb']:>12.1f} {r['avg_time_ms']:>12.2f} {'OK':>10}")

    # Create summary table for visualization - include OOM rows
    # Log table: X=num_patches, Y=avg_time_ms, grouped by model
    table_data = []
    for r in all_results:
        if r.get("oom"):
            # Include OOM rows with None for metrics
            table_data.append([model_name, r["num_patches"], None, None, "OOM"])
        else:
            table_data.append([model_name, r["num_patches"], r["avg_time_ms"], r["peak_vram_mb"], "OK"])

    table = wandb.Table(
        data=table_data,
        columns=["model", "num_patches", "avg_time_ms", "peak_vram_mb", "status"]
    )
    wandb.log({"benchmark_results": table})

    # Only create plots if we have valid results
    valid_results = [r for r in all_results if not r.get("oom")]
    if valid_results:
        valid_table = wandb.Table(
            data=[
                [model_name, r["num_patches"], r["avg_time_ms"], r["peak_vram_mb"]]
                for r in valid_results
            ],
            columns=["model", "num_patches", "avg_time_ms", "peak_vram_mb"]
        )

        # Log time vs num_patches line/scatter plot
        wandb.log({
            "time_vs_patches": wandb.plot.line(
                valid_table, "num_patches", "avg_time_ms",
                title="Time vs Patch Count",
                stroke="model"
            ),
        })

        # Also log VRAM vs patches
        wandb.log({
            "vram_vs_patches": wandb.plot.line(
                valid_table, "num_patches", "peak_vram_mb",
                title="VRAM vs Patch Count",
                stroke="model"
            ),
        })

        # Store summary metrics
        for r in valid_results:
            wandb.summary[f"vram_n{r['num_patches']}"] = r["peak_vram_mb"]
            wandb.summary[f"time_n{r['num_patches']}"] = r["avg_time_ms"]

    # Mark OOM boundary - log last successful N as metric
    oom_results = [r for r in all_results if r.get("oom")]
    if oom_results:
        first_oom = min(r["num_patches"] for r in oom_results)
        wandb.summary["first_oom_n"] = first_oom
        wandb.summary["max_successful_n"] = max((r["num_patches"] for r in valid_results), default=0)
        print(f"\nOOM at N={first_oom:,} (last successful: N={wandb.summary['max_successful_n']:,})")

    wandb.finish()
    print(f"\nResults logged to wandb project: {args.project}")


if __name__ == "__main__":
    main()
