"""Evaluation helpers for the Week 1 Branch A baseline."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score


def compute_binary_classification_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    average_loss: float,
) -> Dict[str, float]:
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "loss": float(average_loss),
    }
