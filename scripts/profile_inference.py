"""Profile Phase 3 inference latency by branch and full fusion.

The profiler uses synthetic 64x64 tensors so the result isolates model forward
latency from image decoding and dataloader throughput. By default it runs CPU
and the first available accelerator, then writes a JSON summary and Markdown
table under ``runs/inference_profile``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.celeba_loader import load_config
from models.discriminator import DiscriminatorPhase3, load_phase3_checkpoint


def _resolve_requested_devices(names: Iterable[str]) -> list[torch.device]:
    devices: list[torch.device] = []
    for name in names:
        if name == "auto":
            devices.append(torch.device("cpu"))
            if torch.backends.mps.is_available():
                devices.append(torch.device("mps"))
            elif torch.cuda.is_available():
                devices.append(torch.device("cuda"))
            continue
        if name == "mps":
            if torch.backends.mps.is_available():
                devices.append(torch.device("mps"))
            else:
                print("WARNING: MPS unavailable; skipping", flush=True)
            continue
        if name == "cuda":
            if torch.cuda.is_available():
                devices.append(torch.device("cuda"))
            else:
                print("WARNING: CUDA unavailable; skipping", flush=True)
            continue
        devices.append(torch.device("cpu"))

    unique: list[torch.device] = []
    seen: set[str] = set()
    for device in devices:
        key = str(device)
        if key not in seen:
            unique.append(device)
            seen.add(key)
    return unique


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _profile_operation(
    *,
    name: str,
    operation: Callable[[], torch.Tensor | dict[str, torch.Tensor]],
    batch_size: int,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, float | int | str]:
    with torch.inference_mode():
        for _ in range(warmup):
            operation()
        _synchronize(device)

        elapsed_ms: list[float] = []
        for _ in range(iterations):
            start = time.perf_counter()
            operation()
            _synchronize(device)
            elapsed_ms.append((time.perf_counter() - start) * 1000.0)

    mean_batch_ms = statistics.fmean(elapsed_ms)
    median_batch_ms = statistics.median(elapsed_ms)
    return {
        "name": name,
        "iterations": iterations,
        "batch_size": batch_size,
        "mean_batch_ms": round(mean_batch_ms, 4),
        "median_batch_ms": round(median_batch_ms, 4),
        "mean_ms_per_image": round(mean_batch_ms / batch_size, 4),
        "median_ms_per_image": round(median_batch_ms / batch_size, 4),
    }


def _load_model(config_path: Path, checkpoint_path: Path, device: torch.device) -> DiscriminatorPhase3:
    config = dict(load_config(config_path))
    phase3_cfg = dict(config.get("phase3", {}))
    model = DiscriminatorPhase3(dropout=float(phase3_cfg.get("dropout", 0.3)))
    load_phase3_checkpoint(model, None, None, checkpoint_path)
    return model.eval().to(device)


def _profile_device(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    model = _load_model(Path(args.config), Path(args.checkpoint), device)
    frame_a = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    frame_b = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    flow = torch.randn(args.batch_size, 2, args.image_size, args.image_size, device=device)

    operations: list[tuple[str, Callable[[], torch.Tensor | dict[str, torch.Tensor]]]] = [
        ("branch_a", lambda: model.branch_a(frame_a)),
        ("branch_b", lambda: model.branch_b(frame_a, frame_b)),
        ("branch_c", lambda: model.branch_c(frame_a, frame_b, flow)),
        ("full_fusion", lambda: model.forward_with_branch_features(frame_a, frame_b, flow)),
    ]
    results = [
        _profile_operation(
            name=name,
            operation=operation,
            batch_size=args.batch_size,
            device=device,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        for name, operation in operations
    ]
    return {"device": str(device), "operations": results}


def _write_markdown(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Inference Profile",
        "",
        f"Checkpoint: `{summary['checkpoint']}`",
        f"Batch size: `{summary['batch_size']}` | Image size: `{summary['image_size']}`",
        "",
        "| Device | Operation | Mean ms/image | Median ms/image | Mean ms/batch |",
        "| ------ | --------- | ------------: | --------------: | ------------: |",
    ]
    for device_result in summary["devices"]:  # type: ignore[index]
        device = device_result["device"]
        for operation in device_result["operations"]:
            lines.append(
                f"| {device} | {operation['name']} "
                f"| {operation['mean_ms_per_image']:.4f} "
                f"| {operation['median_ms_per_image']:.4f} "
                f"| {operation['mean_batch_ms']:.4f} |"
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile Phase 3 inference latency")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/phase3_a_b_c.pt")
    parser.add_argument("--run-dir", default="runs/inference_profile")
    parser.add_argument("--device", action="append", default=["auto"], choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    devices = _resolve_requested_devices(args.device)
    if not devices:
        raise RuntimeError("No requested profiling devices are available")

    summary: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "devices": [_profile_device(args, device) for device in devices],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(run_dir / "summary.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:])
