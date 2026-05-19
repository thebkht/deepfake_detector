"""Standalone Branch A checkpoint evaluation on a held-out split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, cast

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from data.celeba_loader import create_celeba_dataloader, load_config
from evaluation.metrics import compute_binary_classification_metrics
from models import BranchABaseline


def _as_str_key_mapping(value: object, *, context: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping for {context}")
    return {str(key): item for key, item in value.items()}


def _resolve_device(device_override: Optional[str] = None) -> torch.device:
    requested = device_override.lower() if device_override is not None else None

    if requested is not None:
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Requested device 'cuda' but CUDA is not available")
            return torch.device("cuda")
        if requested == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("Requested device 'mps' but MPS is not available")
            return torch.device("mps")
        if requested == "cpu":
            return torch.device("cpu")
        raise ValueError(f"Unsupported device override: {device_override}")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _run_evaluation_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            frame_a = batch["frame_a"].to(device)
            frame_b = batch["frame_b"].to(device)
            labels = batch["label"].float().to(device)

            logits = model(frame_a, frame_b)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_examples += batch_size
            total_loss += loss.item() * batch_size
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    if total_examples == 0:
        raise ValueError("Received an empty dataloader split; cannot compute evaluation metrics")

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels).astype(np.int64)
    average_loss = total_loss / total_examples
    metrics = compute_binary_classification_metrics(logits=logits, labels=labels, average_loss=average_loss)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    predictions = (probabilities >= 0.5).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "metrics": metrics,
        "confusion_matrix": {
            "tn": int(matrix[0, 0]),
            "fp": int(matrix[0, 1]),
            "fn": int(matrix[1, 0]),
            "tp": int(matrix[1, 1]),
        },
        "num_examples": int(total_examples),
    }


def _write_eval_reports(
    run_dir: Path,
    *,
    split: str,
    checkpoint_path: Path,
    device: torch.device,
    results: Dict[str, Any],
) -> None:
    payload = {
        "split": split,
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "num_examples": results["num_examples"],
        "metrics": results["metrics"],
        "confusion_matrix": results["confusion_matrix"],
    }
    json_path = run_dir / "branch_a_test_confusion_matrix.json"
    md_path = run_dir / "branch_a_test_confusion_matrix.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# Branch A Test Split Evaluation",
                "",
                f"- Split: `{split}`",
                f"- Checkpoint: `{checkpoint_path}`",
                f"- Device: `{device}`",
                f"- Num examples: `{payload['num_examples']}`",
                f"- Balanced accuracy: `{payload['metrics']['balanced_accuracy']:.4f}`",
                f"- F1: `{payload['metrics']['f1']:.4f}`",
                f"- Loss: `{payload['metrics']['loss']:.4f}`",
                "",
                "## Confusion Matrix",
                "",
                f"- TN: `{payload['confusion_matrix']['tn']}`",
                f"- FP: `{payload['confusion_matrix']['fp']}`",
                f"- FN: `{payload['confusion_matrix']['fn']}`",
                f"- TP: `{payload['confusion_matrix']['tp']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def evaluate_branch_a_checkpoint(
    config_path: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    run_name: str = "branch_a_test_eval",
    split: str = "test",
    limit: Optional[int] = None,
    device_override: Optional[str] = None,
) -> Dict[str, Any]:
    if split.lower() != "test":
        raise ValueError("Branch A standalone evaluator currently supports the test split only")

    config = _as_str_key_mapping(load_config(config_path), context="config")
    paths_cfg = _as_str_key_mapping(config["paths"], context="config.paths")
    training_cfg = _as_str_key_mapping(config["training"], context="config.training")
    resolved_checkpoint = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else Path(str(paths_cfg["checkpoints_dir"])) / str(training_cfg["checkpoint_name"])
    )
    if not resolved_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {resolved_checkpoint}")

    device = _resolve_device(device_override)
    model = BranchABaseline(dropout=float(training_cfg["dropout"])).to(device)
    checkpoint = torch.load(resolved_checkpoint, map_location=device, weights_only=False)
    state_dict = cast(dict[str, torch.Tensor], checkpoint["model_state_dict"])
    model.load_state_dict(state_dict)
    criterion = nn.BCEWithLogitsLoss()

    dataloader = create_celeba_dataloader(config, split=split, shuffle=False, limit=limit)
    results = _run_evaluation_epoch(model, dataloader, criterion, device)

    run_dir = Path(str(paths_cfg["runs_dir"])) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_eval_reports(
        run_dir,
        split=split,
        checkpoint_path=resolved_checkpoint,
        device=device,
        results=results,
    )
    return {
        "run_dir": str(run_dir),
        "split": split,
        "checkpoint_path": str(resolved_checkpoint),
        "device": str(device),
        "num_examples": results["num_examples"],
        "metrics": results["metrics"],
        "confusion_matrix": results["confusion_matrix"],
    }
