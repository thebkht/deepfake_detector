# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A hybrid three-branch deepfake face detector trained in sequential phases on CelebA. **Important framing:** the current task is a *proxy* — "same identity vs. different identity" image pairs, not real generative deepfakes. In-domain proxy metrics look strong (~0.88 balanced accuracy) but all out-of-domain forensics transfer runs currently *fail* the deployment gate (~0.47 balanced accuracy). Treat proxy numbers as a baseline, never as a deepfake result. `README.md` is the authoritative, detailed status log; this file is the orientation map.

## Commands

All commands run from the repo root with the project venv active. Modules are invoked with `python3 -m`; entry points live under `training/`, `evaluation/`, `scripts/`, and `data/`.

```bash
# Run the full test suite (unittest, not pytest — there is no pytest config)
python -m unittest discover -s tests
# Single test module / case
python -m unittest tests.test_model
python -m unittest tests.test_architecture_contracts.TestName.test_method

# Phased training — each phase loads the previous phase's checkpoint
python3 -m training.phase1_train --config config/config.yaml --run-name branch_a_baseline
python3 -m training.phase2_train --config config/config.yaml --run-name phase2_a_b
python3 -m training.phase3_train --config config/config.yaml --run-name phase3_a_b_c
python3 -m training.phase4_finetune --config config/config.yaml --run-name phase4_ensemble

# Optical-flow cache (required substrate for Branch C / Phase 3+)
python3 -m data.precompute_flow --img-dir data/celeba/img_align_celeba \
  --out-dir data/flow_cache --method farneback --image-size 64

# Evaluation
python3 -m training.eval_branch_a --config config/config.yaml --run-name branch_a_test_eval
python3 scripts/run_ensemble_ablation.py --config config/config.yaml \
  --checkpoint checkpoints/phase3_a_b_c.pt --run-dir runs/ensemble_ablation --device cpu
python3 -m evaluation.threshold_sweep --config config/config.yaml \
  --checkpoint checkpoints/phase3_a_b_c.pt --run-dir runs/threshold_sweep_phase3 --device cpu
python3 scripts/run_forensics_eval.py --config config/config.yaml \
  --checkpoint checkpoints/phase3_a_b_c.pt --device cpu --dataset "Data Set 1" --limit 128 --max-batches 4
```

Common smoke-run flags across trainers: `--train-limit`, `--val-limit`, `--epochs-override`, `--max-batches`, `--checkpoint-name-override` (avoid clobbering real checkpoints), `--device cpu|cuda|mps`. Device default resolves `mps → cuda → cpu` (Phase 3/4 read `phaseN.device` first).

## Environment

- Three venvs exist. `.venv-mps/` is the working one (has `facenet-pytorch` + MPS torch for the face-alignment path); `.venv/` may be stale. `requirements.txt` is the source of truth.
- Data and outputs are gitignored and not committed: `data/celeba/`, `data/forensics/`, `data/flow_cache/`, `checkpoints/`, `runs/`. Expect them absent on a fresh clone — CelebA comes from `scripts/download_celeba.sh` (needs Kaggle CLI + `~/.kaggle/kaggle.json`).
- This repo has a CodeGraph MCP index (`.codegraph/`); prefer `codegraph_*` tools for structural lookups (see global instructions).

## Architecture

A four-phase additive pipeline. Each phase reuses and freezes earlier branches, trains only the new piece plus a fresh fusion head, and writes a checkpoint the next phase loads. Models in `models/`, trainers in `training/phaseN_trainer.py` (the `phaseN_train.py` files are thin CLI wrappers).

- **Branch A** (`models/branch_a.py`) — spatial CNN encoder, 5 spectral-norm conv blocks → **2048-D** per frame. `BranchABaseline` classifies the concatenated 4096-D pair. This is Phase 1.
- **Branch B** (`models/branch_b.py`) — reuses the pretrained `BranchAEncoder` on both frames, computes an **8-D** temporal/velocity summary, then `LayerNorm(8)` and expands it to a learned **32-D** feature. Phase 2 = `DiscriminatorPhase2`, fuses `2048 + 32 = 2080`.
- **Branch C** (`models/branch_c.py`) — deterministic **28-D** extractor (20-D optical-flow summaries + 8-D HSV photometric). Phase 3 = `DiscriminatorPhase3`, fuses `2048 + 32 + 28 = 2108`.
- **Phase 4** (`DiscriminatorPhase4`) — staged-unfreeze fine-tune of the 2108-D stack with `AsymmetricCombinedLoss` (upweights real-class errors).

### Critical contracts and gotchas

- **Active fusion contract is `2048 + 32 + 28 = 2108`, NOT the proposal's `2048 + 8 + 28 = 2084`.** The 32 (not 8) comes from Branch B's learned expansion. Checkpoints are load-compatible only with 2108. Don't "fix" this to match the proposal without retraining.
- **Phase 3/4 must use `pairing_mode="adjacent_cache"`.** Cached flow tensors are keyed to adjacent frame pairs; switching to default pairing silently attaches the wrong flow to a pair unless the cache is regenerated. Phase 3 preflight verifies the flow cache stem set before epoch 0 and requires `include_flow=True`.
- **Phase 3 is the current deployment candidate, not Phase 4.** Phase 4's asymmetric loss raised fake recall but worsened real-class TNR and F1 (overfit the proxy distribution).
- **`checkpoints/phase2_a_b.pt` may be a legacy pre-"Run 3" checkpoint** on an older Branch B architecture; verify provenance before using it as a Phase 3 base.
- **Label convention is fake-positive** (fake = 1). The hinge/asymmetric losses in `training/losses.py` depend on this; preserve it.

### Supporting layers

- `data/celeba_loader.py` — builds real/fake pairs. Uses `identity_CelebA.txt` when present (same-identity = real, cross-identity = fake); falls back to adjacent / deterministic-distant index pairing when absent. The checked-in identity file may be attribute-derived pseudo-identities, not true CelebA labels.
- `data/face_align.py` + `data/forensics_loader.py` — MTCNN alignment (facenet-pytorch) with center-crop fallback, used by the forensics OOD path.
- `training/` shared infra: `checkpointing.py` (save/load/resume + remap), `losses.py`, `overfit_stop.py` (early stop: val-loss-worsens-while-train-improves trend + a phase-specific val-loss ceiling), `run_artifacts.py` (preview grids, confusion matrices, `results.png`), `tracker.py` (optional TensorBoard).
- `evaluation/` — `eval.py` (balanced acc / F1 / AUC-ROC / confusion plots), `ensemble.py` (extracts frozen A/B/C features for the 7 canonical branch-combination RF/logistic probes), `threshold_sweep.py`, `forensics_eval.py`/`ood_eval.py`.

## Config

`config/config.yaml` holds per-phase blocks (`training`/Phase 1, `phase2`, `phase3`, `phase4`, plus `forensics`). Each block defines its own epochs, LR, optimizer/scheduler, checkpoint name, pretrained-input checkpoint, targets, and early-stopping thresholds. Checkpoint selection metric is `balanced_accuracy`. When changing training behavior, edit the relevant phase block — trainers read from it, and CLI flags override per-run.
