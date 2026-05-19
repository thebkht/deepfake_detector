from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
import yaml
from PIL import Image

from evaluation import compute_binary_classification_metrics
from evaluation.branch_a_eval import evaluate_branch_a_checkpoint
from models import BranchABaseline
from training.branch_a_trainer import _build_scheduler, train_branch_a


class BranchABaselineTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.image_root = self.root / "celeba" / "img_align_celeba"
        self.image_root.mkdir(parents=True, exist_ok=True)
        self._write_sample_images(count=24)
        self.config_path = self.root / "config.yaml"
        config = {
            "paths": {
                "project_root": ".",
                "image_dir": str(self.image_root),
                "identity_file": str(self.root / "celeba" / "missing_identity.txt"),
                "checkpoints_dir": str(self.root / "checkpoints"),
                "runs_dir": str(self.root / "runs"),
            },
            "dataset": {
                "image_size": 64,
                "image_channels": 3,
                "fake_ratio": 0.5,
                "train_split": 0.8,
                "val_split": 0.1,
                "test_split": 0.1,
                "gaussian_noise_std": 0.05,
                "expected_image_count": 24,
                "expected_native_width": 178,
                "expected_native_height": 218,
            },
            "dataloader": {
                "batch_size": 4,
                "num_workers": 0,
                "pin_memory": False,
                "drop_last": False,
            },
            "training": {
                "epochs": 1,
                "learning_rate": 0.0002,
                "optimizer": "adam",
                "betas": [0.5, 0.999],
                "scheduler": "CosineAnnealingLR",
                "scheduler_t_max": 100,
                "dropout": 0.3,
                "checkpoint_metric": "balanced_accuracy",
                "seed": 42,
                "checkpoint_name": "phase1_branch_a_best.pt",
            },
        }
        self.config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_sample_images(self, count: int) -> None:
        for idx in range(1, count + 1):
            image = Image.new("RGB", (178, 218), color=(idx * 7 % 255, idx * 11 % 255, idx * 13 % 255))
            image.save(self.image_root / f"{idx:06d}.jpg")

    def test_branch_a_forward_output_shape(self) -> None:
        model = BranchABaseline()
        frame_a = torch.randn(2, 3, 64, 64)
        frame_b = torch.randn(2, 3, 64, 64)
        logits = model(frame_a, frame_b)
        self.assertEqual(tuple(logits.shape), (2,))

    def test_metric_computation_sanity(self) -> None:
        metrics = compute_binary_classification_metrics(
            logits=torch.tensor([2.0, -2.0, 3.0, -3.0]).numpy(),
            labels=torch.tensor([1, 0, 1, 0]).numpy(),
            average_loss=0.1,
        )
        self.assertAlmostEqual(metrics["balanced_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 1.0)
        self.assertAlmostEqual(metrics["loss"], 0.1)

    def test_scheduler_uses_configured_t_max(self) -> None:
        model = BranchABaseline()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
        scheduler = _build_scheduler(
            optimizer,
            {"scheduler": "CosineAnnealingLR", "scheduler_t_max": 100},
        )
        self.assertEqual(scheduler.T_max, 100)

    def test_training_smoke_writes_checkpoint_and_summary(self) -> None:
        summary = train_branch_a(self.config_path, run_name="smoke")
        checkpoint_path = self.root / "checkpoints" / "phase1_branch_a_best.pt"
        summary_path = self.root / "runs" / "smoke" / "benchmark_summary.json"
        history_path = self.root / "runs" / "smoke" / "metrics_history.json"
        confusion_path = self.root / "runs" / "smoke" / "confusion_matrix.png"
        confusion_norm_path = self.root / "runs" / "smoke" / "confusion_matrix_normalized.png"
        results_path = self.root / "runs" / "smoke" / "results.png"

        self.assertTrue(checkpoint_path.exists())
        self.assertTrue(summary_path.exists())
        self.assertTrue(history_path.exists())
        self.assertTrue(confusion_path.exists())
        self.assertTrue(confusion_norm_path.exists())
        self.assertTrue(results_path.exists())
        self.assertIn(summary["status"], {"met", "partially met", "not met"})
        self.assertEqual(summary["hyperparameters"]["scheduler_t_max"], 100)

        saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(saved_summary["checkpoint_selection_rule"], "highest validation balanced accuracy")
        self.assertIn("cross-identity proxy negatives", saved_summary["limitations"][1])

    def test_branch_a_test_eval_writes_confusion_matrix_report(self) -> None:
        train_branch_a(self.config_path, run_name="smoke_eval")
        results = evaluate_branch_a_checkpoint(
            self.config_path,
            run_name="branch_a_test_eval",
            split="test",
            device_override="cpu",
        )
        run_dir = self.root / "runs" / "branch_a_test_eval"
        json_path = run_dir / "confusion_matrix.json"
        png_path = run_dir / "confusion_matrix.png"
        md_path = run_dir / "eval_report.md"

        self.assertEqual(results["split"], "test")
        self.assertTrue(json_path.exists())
        self.assertTrue(png_path.exists())
        self.assertTrue(md_path.exists())
        saved_results = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(set(saved_results["confusion_matrix"].keys()), {"tn", "fp", "fn", "tp"})
        self.assertIn("auc_roc", saved_results)
