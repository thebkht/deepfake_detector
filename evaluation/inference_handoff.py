"""Inference handoff artifact helpers for the Phase 4 checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping


def write_inference_contract(
    run_dir: str | Path,
    *,
    checkpoint_path: str | Path,
    fusion_contract: str,
    branch_dims: Mapping[str, int],
    pairing_mode: str,
    include_flow: bool,
    recommended_combo: str,
    recommended_combo_gate_met: bool,
) -> Path:
    run_path = Path(run_dir)
    artifact_path = run_path / "inference_contract.json"
    payload: Dict[str, Any] = {
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "fusion_contract": fusion_contract,
        "fusion_dim": int(sum(int(value) for value in branch_dims.values())),
        "branch_dims": {str(key): int(value) for key, value in branch_dims.items()},
        "normalization": {
            "image_range": [-1.0, 1.0],
            "image_size": [64, 64],
            "flow_shape": [2, 64, 64],
        },
        "pairing_mode": pairing_mode,
        "include_flow": bool(include_flow),
        "recommended_combo": recommended_combo,
        "recommended_combo_gate_met": bool(recommended_combo_gate_met),
        "week4_cli_stub": (
            "python -m evaluation.ood_eval "
            f"--config config/config.yaml --checkpoint {Path(checkpoint_path).name} "
            "--image-dir <week4_eval_dir>"
        ),
    }
    artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return artifact_path


__all__ = ["write_inference_contract"]
