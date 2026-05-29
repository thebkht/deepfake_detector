from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path

import torch

from models.branch_b import BranchB_Spatiotemporal
from models.branch_c import BranchC_Physics
from models.discriminator import DiscriminatorPhase3, DiscriminatorPhase4, FUSION_DIM_2108
from training.losses import AsymmetricCombinedLoss, HingeLoss


class ArchitectureContractTestCase(unittest.TestCase):
    def _inputs(self, batch_size: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(7)
        frame_a = torch.randn(batch_size, 3, 64, 64)
        frame_b = torch.randn(batch_size, 3, 64, 64)
        flow = torch.randn(batch_size, 2, 64, 64)
        return frame_a, frame_b, flow

    def test_phase3_forward_contract_and_branch_wiring(self) -> None:
        model = DiscriminatorPhase3().eval()
        frame_a, frame_b, flow = self._inputs()

        with torch.no_grad():
            outputs = model.forward_with_branch_features(frame_a, frame_b, flow)
            logits = model(frame_a, frame_b, flow)

        self.assertEqual(set(outputs), {"a", "b", "c", "logit"})
        self.assertEqual(tuple(outputs["a"].shape), (2, 2048))
        self.assertEqual(tuple(outputs["b"].shape), (2, 32))
        self.assertEqual(tuple(outputs["c"].shape), (2, 28))
        self.assertEqual(tuple(outputs["logit"].shape), (2,))
        self.assertTrue(torch.equal(outputs["logit"], logits))
        self.assertEqual(model.fusion_dim, FUSION_DIM_2108)

    def test_phase4_forward_contract_and_branch_wiring(self) -> None:
        model = DiscriminatorPhase4().eval()
        frame_a, frame_b, flow = self._inputs()

        with torch.no_grad():
            outputs = model.forward_with_branch_features(frame_a, frame_b, flow)
            logits = model(frame_a, frame_b, flow)

        self.assertEqual(set(outputs), {"a", "b", "c", "logit"})
        self.assertEqual(tuple(outputs["a"].shape), (2, 2048))
        self.assertEqual(tuple(outputs["b"].shape), (2, 32))
        self.assertEqual(tuple(outputs["c"].shape), (2, 28))
        self.assertEqual(tuple(outputs["logit"].shape), (2,))
        self.assertTrue(torch.equal(outputs["logit"], logits))
        self.assertEqual(sum(model.branch_dims.values()), FUSION_DIM_2108)

    def test_safe_normalization_guards_are_present(self) -> None:
        branch_b_source = inspect.getsource(BranchB_Spatiotemporal._summary_features)
        branch_c_source = inspect.getsource(BranchC_Physics._flow_summary)

        self.assertIn("eps=1e-8", branch_b_source)
        self.assertIn("min=1e-12", branch_c_source)

        branch_c = BranchC_Physics()
        frame_a = torch.zeros(2, 3, 64, 64)
        frame_b = torch.zeros(2, 3, 64, 64)
        flow = torch.zeros(2, 2, 64, 64)
        output = branch_c(frame_a, frame_b, flow)
        self.assertFalse(torch.isnan(output).any())

    def test_losses_are_finite_on_random_batch(self) -> None:
        logits = torch.randn(8)
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])

        asymmetric = AsymmetricCombinedLoss()(logits, labels)
        hinge = HingeLoss()(logits, labels)

        self.assertTrue(torch.isfinite(asymmetric))
        self.assertTrue(torch.isfinite(hinge))

    def test_phase4_inference_contract_sums_to_2108(self) -> None:
        contract_path = Path("runs/phase4_ensemble/inference_contract.json")
        self.assertTrue(contract_path.exists(), msg=f"Missing {contract_path}")
        contract = json.loads(contract_path.read_text(encoding="utf-8"))

        self.assertEqual(contract["fusion_contract"], "2108")
        self.assertEqual(contract["fusion_dim"], 2108)
        self.assertEqual(sum(contract["branch_dims"].values()), 2108)
        self.assertEqual(contract["normalization"]["flow_shape"], [2, 64, 64])


if __name__ == "__main__":
    unittest.main()
