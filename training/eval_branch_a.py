"""CLI entrypoint for standalone Branch A checkpoint evaluation."""

from __future__ import annotations

import argparse

from evaluation.branch_a_eval import evaluate_branch_a_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Branch A checkpoint on the test split")
    parser.add_argument("--config", default="config/config.yaml", help="Path to the YAML config")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path. Defaults to the configured Branch A checkpoint.",
    )
    parser.add_argument("--run-name", default="branch_a_test_eval", help="Run directory name")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for test samples")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        default=None,
        help="Optional device override. Defaults to cuda, then mps, then cpu.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluate_branch_a_checkpoint(
        args.config,
        checkpoint_path=args.checkpoint,
        run_name=args.run_name,
        split="test",
        limit=args.limit,
        device_override=args.device,
    )


if __name__ == "__main__":
    main()
