"""CLI entrypoint for Week 3 Phase 4 fine-tuning."""

from __future__ import annotations

import argparse

from training.phase4_trainer import train_phase4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Week 3 Phase 4 end-to-end discriminator")
    parser.add_argument("--config", default="config/config.yaml", help="Path to the YAML config")
    parser.add_argument("--train-limit", type=int, default=None, help="Optional cap for train samples")
    parser.add_argument("--val-limit", type=int, default=None, help="Optional cap for val samples")
    parser.add_argument("--run-name", default="phase4_ensemble", help="Run directory name")
    parser.add_argument("--epochs-override", type=int, default=None, help="Optional epoch override for dry runs")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional cap for batches per split")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional dataloader worker override")
    parser.add_argument("--checkpoint-name-override", default=None, help="Optional checkpoint filename override")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None, help="Optional device override")
    parser.add_argument("--tracker-backend", default=None, help="Optional tracking backend")
    parser.add_argument("--resume", default=None, help="Optional Phase 4 checkpoint to resume from")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_phase4(
        args.config,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        run_name=args.run_name,
        tracker_backend=args.tracker_backend,
        epochs_override=args.epochs_override,
        device_override=args.device,
        max_batches=args.max_batches,
        num_workers_override=args.num_workers,
        checkpoint_name_override=args.checkpoint_name_override,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
