"""Loss modules shared across later training phases."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class HingeLoss(nn.Module):
    """
    Master-plan §8 GAN hinge on discriminator logits.

    Convention (pinned — do not invert in Phase 4):
      - real samples (dataset label 0): loss = max(0, 1 - logit)
      - fake samples (dataset label 1): loss = max(0, 1 + logit)

    Dataset labels remain 0=real, 1=fake (same as BCEWithLogitsLoss).
    """

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        if logits.shape != labels.shape:
            raise ValueError(
                f"HingeLoss expects logits and labels with identical shape, got {tuple(logits.shape)} and {tuple(labels.shape)}"
            )
        labels_long = labels.long()
        real_loss = torch.relu(1.0 - logits)
        fake_loss = torch.relu(1.0 + logits)
        loss = torch.where(labels_long == 0, real_loss, fake_loss)
        return loss.mean()


class CombinedBCEHingeLoss(nn.Module):
    """Phase 4 combined loss with fixed BCE and hinge weights."""

    def __init__(self, bce_weight: float = 0.7, hinge_weight: float = 0.3) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.hinge_weight = float(hinge_weight)
        self.bce = nn.BCEWithLogitsLoss()
        self.hinge = HingeLoss()

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        if logits.shape != labels.shape:
            raise ValueError(
                "CombinedBCEHingeLoss expects logits and labels with identical shape, "
                f"got {tuple(logits.shape)} and {tuple(labels.shape)}"
            )
        bce_loss = self.bce(logits, labels)
        hinge_loss = self.hinge(logits, labels)
        return (self.bce_weight * bce_loss) + (self.hinge_weight * hinge_loss)


class AsymmetricCombinedLoss(nn.Module):
    """Fake-positive BCE plus hinge loss that upweights real-sample mistakes."""

    def __init__(
        self,
        bce_weight: float = 0.7,
        hinge_weight: float = 0.3,
        real_weight: float = 1.5,
        fake_weight: float = 1.0,
        margin: float = 0.8,
    ) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.hinge_weight = float(hinge_weight)
        self.real_weight = float(real_weight)
        self.fake_weight = float(fake_weight)
        self.margin = float(margin)
        if self.real_weight <= 0 or self.fake_weight <= 0:
            raise ValueError("Class weights must be positive")
        if self.margin <= 0:
            raise ValueError("Margin must be positive")
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        if logits.shape != labels.shape:
            raise ValueError(
                "AsymmetricCombinedLoss expects logits and labels with identical shape, "
                f"got {tuple(logits.shape)} and {tuple(labels.shape)}"
            )

        labels_float = labels.float()
        labels_long = labels.long()
        bce_weights = torch.where(
            labels_long == 0,
            torch.full_like(logits, self.real_weight),
            torch.full_like(logits, self.fake_weight),
        )
        bce_loss = (self.bce(logits, labels_float) * bce_weights).mean()

        real_mask = labels_long == 0
        fake_mask = labels_long == 1
        hinge_terms: list[Tensor] = []
        if real_mask.any():
            # Positive logits mean fake, so real samples should sit below -margin.
            hinge_terms.append(torch.relu(self.margin + logits[real_mask]).mean() * self.real_weight)
        if fake_mask.any():
            hinge_terms.append(torch.relu(self.margin - logits[fake_mask]).mean() * self.fake_weight)
        hinge_loss = torch.stack(hinge_terms).sum() if hinge_terms else logits.sum() * 0.0

        return (self.bce_weight * bce_loss) + (self.hinge_weight * hinge_loss)


__all__ = ["HingeLoss", "CombinedBCEHingeLoss", "AsymmetricCombinedLoss"]
