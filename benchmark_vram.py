"""VRAM benchmarking script for MIL models.

Tests multiple patch counts (N) and logs VRAM usage vs time to wandb.

Usage:
    python benchmark_vram.py --config configs/final_camely_abmil.py
    python benchmark_vram.py --config configs/final_camely_abmil.py --patch-counts 10000 50000 100000
"""

import argparse
import time
import torch
import torch.multiprocessing as mp
from pathlib import Path
import wandb
import os
import json

from wsi_classification.experiments.utils.cli import load_config_from_file, apply_config_overrides
from wsi_classification.experiments.datamodules.h5_datamodule import H5FeatureBagDataModule


# Set multiprocessing start method for CUDA compatibility
mp.set_start_method('spawn', force=True)


def run_benchmark_subprocess(config_path, num_patches, feature_dim, warmup_steps, measure_steps, overrides, result_queue):
    """Run a single benchmark in a subprocess."""
    try:
        import torch
        import time
        from pathlib import Path
        from wsi_classification.experiments.default_cfg import ExperimentConfig
        from wsi_classification.experiments.utils.cli import load_config_from_file, apply_config_overrides
        from wsi_classification.experiments.utils.lazy_config import instantiate
        from wsi_classification.experiments.datamodules.h5_datamodule import H5FeatureBagDataModule

        # Load config
        config = load_config_from_file(config_path)
        if overrides:
            config = apply_config_overrides(config, overrides)

        # Create datamodule
        datamodule = H5FeatureBagDataModule(
            train_csv="dummy.csv",
            val_csv="dummy.csv",
            use_synthetic=True,
            synthetic_num_patches=num_patches,
            synthetic_feature_dim=feature_dim,
            batch_size=1,
            num_workers=0,
        )
        datamodule.prepare_data()
        datamodule.setup("fit")

        # Build model
        network = instantiate(config.net, in_features=feature_dim, out_features=1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = instantiate(config.lightning_wrapper_class, network=network, cfg=config)
        model = model.to(device)
        model.train()

        optimizer = instantiate(config.optimizer, params=model.parameters())
        train_loader = datamodule.train_dataloader()
        data_iter = iter(train_loader)

        # Warmup
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

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Measure
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

            times.append((end - start) * 1000)

        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        avg_time_ms = sum(times) / len(times)

        result_queue.put({
            "success": True,
            "num_patches": num_patches,
            "peak_vram_mb": peak_vram_mb,
            "avg_time_ms": avg_time_ms,
            "min_time_ms": min(times),
            "max_time_ms": max(times),
        })
    except Exception as e:
        result_queue.put({
            "success": False,
            "num_patches": num_patches,
            "error": str(e),
        })


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


def main():
    args = parse_args()

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

    # Run benchmarks in subprocess so OOM kills don't stop main process
    all_results = []
    for num_patches in args.patch_counts:
        print(f"\n{'='*60}")
        print(f"Benchmarking N={num_patches:,} patches, D={args.feature_dim}")
        print(f"{'='*60}")

        result_queue = mp.Queue()
        process = mp.Process(
            target=run_benchmark_subprocess,
            args=(args.config, num_patches, args.feature_dim, args.warmup_steps, args.measure_steps, args.overrides, result_queue)
        )
        process.start()
        process.join(timeout=300)  # 5 minute timeout per benchmark

        if process.is_alive():
            # Timeout
            process.terminate()
            process.join()
            print(f"Timeout at N={num_patches}")
            result = {"num_patches": num_patches, "oom": True}
        elif process.exitcode != 0:
            # Process was killed (OOM) or crashed
            print(f"Process killed/crashed at N={num_patches} (exit code: {process.exitcode})")
            result = {"num_patches": num_patches, "oom": True}
        else:
            # Got result
            try:
                subprocess_result = result_queue.get_nowait()
                if subprocess_result.get("success"):
                    result = {
                        "num_patches": subprocess_result["num_patches"],
                        "peak_vram_mb": subprocess_result["peak_vram_mb"],
                        "avg_time_ms": subprocess_result["avg_time_ms"],
                        "min_time_ms": subprocess_result["min_time_ms"],
                        "max_time_ms": subprocess_result["max_time_ms"],
                    }
                    print(f"Results: VRAM={result['peak_vram_mb']:.1f} MB, Time={result['avg_time_ms']:.2f} ms")
                    wandb.log({
                        "num_patches": num_patches,
                        "peak_vram_mb": result["peak_vram_mb"],
                        "avg_time_ms": result["avg_time_ms"],
                    })
                else:
                    print(f"Failed at N={num_patches}: {subprocess_result.get('error', 'Unknown error')}")
                    result = {"num_patches": num_patches, "oom": True}
            except Exception as e:
                print(f"Failed to get result at N={num_patches}: {e}")
                result = {"num_patches": num_patches, "oom": True}

        all_results.append(result)
        torch.cuda.empty_cache()
        # Small delay to let system recover
        time.sleep(1)

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
