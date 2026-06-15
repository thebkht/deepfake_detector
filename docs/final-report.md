# Final Report: Forensics OOD Evaluation

## Executive Summary

The locked forensics OOD evaluation is complete across all four local test datasets, totaling `20,905` images. The canonical run is `runs/forensics_eval_final/summary.json`, using aligned validation/test faces, degenerate pairing, per-dataset validation thresholds, and Branch B inversion. The result is a clear negative finding: the CelebA-trained neural model and all seven transfer ensembles fail to generalize to the forensics distribution.

**The proposal target of 94.4% balanced accuracy and F1 ≥ 0.93 on forensics OOD content is not achievable with inference-only fixes (MTCNN alignment, threshold calibration, degenerate pairing, TTA) applied to the CelebA-trained checkpoint. Closing the gap to 94.4% requires training or adaptation on forensics or equivalent manipulated-face data — which is explicitly out of scope for this recovery pass.**

The primary handoff artifact is `runs/forensics_eval_final/summary.json`. Per-image scores and confusion-matrix images are under `runs/forensics_eval_final/per_dataset/`, with pooled confusion matrices under `runs/forensics_eval_final/pooled/`. The earlier `runs/forensics_eval/` artifact is retained as the adjacent-pair baseline, not the canonical locked result.

## Protocol

| Item | Value |
| ---- | ----- |
| Evaluation root | `data/forensics` |
| Split | `test` |
| Pairing | `degenerate` |
| Aligned root | `data/forensics_aligned` |
| Threshold mode | Per-dataset validation thresholds from `runs/forensics_threshold/` |
| Branch B inversion | Enabled |
| Neural checkpoint | `checkpoints/phase3_a_b_c.pt` |
| Phase 4 comparison checkpoint | Not included in locked final run |
| CelebA transfer cache | `runs/celeba_features/phase3_train_adjacent_cache.npz` |
| Output directory | `runs/forensics_eval_final` |

## Validation Calibration

The aligned validation run and threshold sweep are complete:

| Artifact | Key result |
| -------- | ---------- |
| `data/forensics_aligned/alignment_report.json` | Validation detection rates: Data Set 1 `99.40%`, Data Set 2 `99.61%`, Data Set 3 `97.56%`, Data Set 4 `97.18%` |
| `runs/forensics_threshold/summary.json` | Pooled neural best balanced accuracy `0.5178` at threshold `0.06`; per-dataset thresholds `{0.07, 0.04, 0.06, 0.06}` |
| `runs/ensemble_ablation_forensics_val/summary.json` | B+C RF pooled balanced accuracy `0.5000`, F1 `0.0000`, AUC-ROC `0.5505`; A+B+C RF is strongest but still only `0.5066` balanced accuracy |

The aligned test cache is not uniformly above the requested 95% detection bar: Data Set 1 test detection is `91.85%`. This is reported as a protocol caveat and not extrapolated.

## Inference-Only Recovery Harness

This repository now includes the reduced-scope recovery path without forensics training:

| Fix | Artifact / option |
| --- | --- |
| MTCNN-aligned cache | `scripts/preprocess_forensics_faces.py --output-root data/forensics_aligned` |
| Degenerate pairing | default `--pairing degenerate` in `scripts/run_forensics_eval.py` |
| Forensics val thresholds | `python -m evaluation.forensics_threshold_sweep --split validation` |
| Per-dataset thresholds | `runs/forensics_threshold/per_dataset_thresholds.json` with `--threshold-mode per_dataset` |
| Branch B polarity flag | `--branch-b-invert-logits` |
| Horizontal flip TTA | `--tta` |

Final claims must be gated on the measured MTCNN detection rate in `data/forensics_aligned/alignment_report.json`. If any dataset is below 95% detection, report measured lift only and do not extrapolate the optimistic MTCNN trajectory.

## Neural Result

| Dataset | N | Bal Acc @0.5 | Bal Acc @threshold | F1 | AUC-ROC |
| ------- | -: | -----------: | -------------: | --: | ------: |
| Data Set 1 | 5227 | 0.5000 | 0.5000 | 0.6683 | 0.4379 |
| Data Set 2 | 5226 | 0.5000 | 0.5000 | 0.6684 | 0.5063 |
| Data Set 3 | 5226 | 0.5000 | 0.5000 | 0.6684 | 0.4858 |
| Data Set 4 | 5226 | 0.5000 | 0.5000 | 0.6684 | 0.4684 |
| Pooled | 20905 | 0.5000 | 0.5000 | 0.6683 | 0.4757 |

The locked neural Phase 3 checkpoint predicts the fake class for nearly every test image under the calibrated low thresholds, giving random balanced accuracy with high fake-class F1.

## Transfer Ensemble Result

Balanced accuracy:

| Dataset | A | B | C | A+B | A+C | B+C | A+B+C |
| ------- | --: | --: | --: | --: | --: | --: | ----: |
| Data Set 1 | 0.5004 | 0.5000 | 0.4295 | 0.5000 | 0.5028 | 0.5000 | 0.5000 |
| Data Set 2 | 0.5006 | 0.5000 | 0.5049 | 0.5000 | 0.5003 | 0.5000 | 0.5000 |
| Data Set 3 | 0.5004 | 0.5000 | 0.5089 | 0.5000 | 0.4986 | 0.5000 | 0.5000 |
| Data Set 4 | 0.5015 | 0.5000 | 0.4998 | 0.5000 | 0.5007 | 0.5000 | 0.5000 |
| Pooled | 0.5007 | 0.5000 | 0.4858 | 0.5000 | 0.5006 | 0.5000 | 0.5000 |

B+C, the proposal-recommended configuration, reaches only `0.5000` pooled balanced accuracy, `0.6683` F1, and `0.5121` AUC-ROC in the locked run. It does not clear the `0.944` balanced-accuracy or `0.93` F1 target.

## Confusion-Matrix Story

| Config | TN | FP | FN | TP | Failure mode |
| ------ | --: | --: | --: | --: | ------------ |
| A only | 59 | 10354 | 31 | 10461 | Real-class collapse; predicts almost everything fake |
| B only | 4735 | 5678 | 5436 | 5056 | Opposite-bias / polarity mismatch on forensics |
| C only | 10 | 10403 | 18 | 10474 | Real-class collapse; predicts almost everything fake |
| A+B | 3125 | 7288 | 3527 | 6965 | Strong fake bias |
| A+C | 203 | 10210 | 160 | 10332 | Near-total fake prediction |
| B+C | 4384 | 6029 | 5013 | 5479 | Below-random transfer |
| A+B+C | 3195 | 7218 | 3548 | 6944 | Strong fake bias |

The earlier adjacent-pair baseline showed mixed real/fake collapse patterns. The locked degenerate-pair run is simpler: most neural and RF outputs collapse to a single-class decision boundary, so balanced accuracy stays at chance. CelebA-trained transfer ensembles are not detecting general GAN artifacts; they learned proxy boundaries tied to CelebA pair statistics.

## Inference Profile

CPU profiling is complete in `runs/inference_profile/summary.json` using batch size `64`, image size `64`, 10 warmup passes, and 100 measured forward passes. MPS/CUDA was unavailable to this profiler run.

| Operation | Mean ms/image | Median ms/image |
| --------- | ------------: | --------------: |
| Branch A | 4.2835 | 3.7802 |
| Branch B | 7.8244 | 7.5536 |
| Branch C | 0.1576 | 0.1569 |
| Full fusion | 12.2136 | 11.4250 |

## In-Domain RF Status

The full frozen in-domain RF benchmark is still pending. A capped smoke run completed with `--skip-transfer-ensemble`, confirming the forensics-train RF path works without retraining the CelebA-transfer probes first. The aligned cache currently contains validation/test splits only; full aligned in-domain RF requires either generating `data/forensics_aligned/*/train/...` or intentionally using raw forensics train/test images.

## Conclusion

Locked OOD evaluation is complete and the gate fails. The current CelebA-trained feature stack should not be presented as a deployable forensics detector.

The next defensible path is to train or adapt on true manipulated-face data, then rerun the same OOD protocol. Until then, the project result should be reported as a negative transfer finding rather than as a successful reproduction of the proposal's B+C OOD claim.
