# Hybrid Three-Branch GAN Discriminator — Build Plan

> Last updated: 2026-06-03  
> Status: **CelebA proxy training is complete, but the forensics gate is not cleared.** The repository includes trained Phase 1-4 checkpoints, local forensics data, aligned validation/test cache, forensics validation threshold calibration, forensics validation ensemble artifacts, a locked forensics test artifact, and CPU inference profiling. The canonical locked test run in `runs/forensics_eval_final/` uses aligned faces, degenerate pairing, per-dataset validation thresholds, and Branch B inversion. B+C RF reaches only `0.5000` pooled balanced accuracy and `0.6683` F1 on the locked test set, far below the 0.944 / 0.93 proposal gate.

> **2 Engineers · 4 Weeks · OOD Robustness Target: 94.4% balanced accuracy**

> **Dataset directive (team lead, 2026-05-29):** CelebA is training-only. Val and test evaluation must use the Real & Fake Images Dataset for Image Forensics (`shivamardeshna/real-and-fake-images-dataset-for-image-forensics`), mixed at balanced 50/50 real/fake. All phase gate metrics must be re-run against the forensics val set before team-lead handoff. The Branch A and Phase 2 proxy scores (1.0 balanced accuracy) were produced against Gaussian noise duplicates and are not valid deepfake benchmarks.

---

## Overview

| Phase | Week | Focus                        | Gate                                                                                    |
| ----- | ---- | ---------------------------- | --------------------------------------------------------------------------------------- |
| 1     | 1    | Setup + Branch A             | Branch A forensics-val acc ≥ 77%, F1 ≥ 0.70; flow cache complete                        |
| 2     | 2    | Branches B & C (parallel)    | `phase2_a_b.pt` and `phase3_a_b_c.pt` both saved; forensics val scores logged           |
| 3     | 3    | Phase 4 fine-tune + ensemble | Phase 4 characterized; ensemble run complete; B+C gate not cleared                      |
| 4     | 4    | Eval & hardening             | Forensics test-set eval complete; final report written; deployment gate status recorded |

---

## Progress snapshot (living)

| Phase | Where the repo is now                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Next gate                                                                                                                       |
| ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| 1     | CelebA flow cache complete at `data/flow_cache` (202,599 `*_flow.pt` files). Local forensics data is staged under `data/forensics`, and `data/forensics_loader.py` resolves the nested `Data Set N/Data Set N` layout with balanced real/fake loading. MTCNN aligned cache exists for validation/test at `data/forensics_aligned`; validation detection is above 95% for all datasets, but Data Set 1 test detection is `91.85%`, below the 95% target. **Pending:** Branch A forensics-val diagnostic remains documented-only, not a canonical GAN single-image score. | Keep Branch A caveated as a pair diagnostic; do not use it as the deployment metric |
| 2     | Phase 2/3/4 checkpoints exist. Forensics validation threshold calibration is logged in `runs/forensics_threshold/summary.json`: pooled neural best balanced accuracy `0.5178` at threshold `0.06`. Forensics validation ensemble is logged in `runs/ensemble_ablation_forensics_val/summary.json`: B+C RF pooled balanced accuracy `0.5000`, F1 `0.0000`, AUC-ROC `0.5505`; A+B+C RF is strongest but still only `0.5066` balanced accuracy. Locked test is logged in `runs/forensics_eval_final/summary.json`: B+C RF pooled balanced accuracy `0.5000`, F1 `0.6683`, AUC-ROC `0.5121`. CPU inference profile exists at `runs/inference_profile/summary.json`. **Pending:** full in-domain RF benchmark and any neural forensics fine-tuning approval. | Run full in-domain RF only after generating aligned train cache or intentionally using raw train/test; neural fine-tune requires team-lead approval |

Update this table when a gate flips so the plan stays honest for the next work session.

---

## Delivery process

**How this plan is used**

- **Checklists** track scope; **gates** (Overview table) decide when to move to the next week. Do not start Week 3 ensemble work until both `phase2_a_b.pt` and `phase3_a_b_c.pt` exist and clear their gates.
- **Checklist truthfulness:** when a commit materially completes a checklist item, update that checkbox in the same commit. Do not leave completed work unchecked because the broader week gate is still open.
- **Dependency direction:** `data/` → `models/` → `training/` → `evaluation/`. Do not import training scripts from model modules.
- **Vertical slices:** prefer a thin working slice (one branch implemented, wired into a training script, producing a checkpoint) over "all model code first, training later."
- **Freeze discipline:** Branch A conv weights must be verifiably frozen before Phase 2 training begins. Add a unit test that confirms no weight change after an optimizer step. Same rule applies to Branch A + B before Phase 3.
- **Proposal parity discipline:** when docs say "proposal", keep the exact tensor contracts from the project proposal. When code deviates temporarily, call it out explicitly instead of silently rewriting the proposal in documentation.
- **Cache contract:** cached flow filenames are `{frame_a_stem}_flow.pt`, computed against the adjacent-index partner rule used by `data/precompute_flow.py`. The loader's real/fake pair sampling can now diverge from that rule. Before Branch C training, either keep Branch C explicitly on adjacent-index pairing or regenerate the cache for the chosen pairing strategy. Do not silently mix cached flow built for one pairing rule with frame pairs sampled by another.
- **Flow-aware eval contract:** Phase 3/4 evaluation must not switch to default pairing while reusing the adjacent flow cache. Keep `adjacent_cache` and balance the evaluated rows if a 50/50 metric view is needed.
- **Commit discipline:** land implementation in small commits by subsystem. Default split: shared data contract change, then model implementation, then training script, then tests. Each commit should answer: what boundary advanced, what verification was run.
- **Docs move with the stage:** if a stage changes a runtime contract (e.g. `__getitem__` return signature), update this file's Progress snapshot and any affected docstrings in the same commit.
- **Eval dataset discipline:** all gate metrics and team-lead benchmarks must be evaluated against the forensics val set, not the CelebA proxy split. CelebA val may be used for training-loop monitoring only.

**Local verification (before calling a task done)**

- Model forward-pass tests: `python -m pytest tests/test_model.py -v`
- Data loader tests: `python -m unittest tests.test_data -v` — includes `test_flow_precompute_smoke`
- Forensics loader tests: `python -m unittest tests.test_data.ForensicsDatasetTestCase -v`
- Overfit-stop tests: `python -m unittest tests.test_overfit_stop -v`
- Branch A eval tests: `python -m unittest tests.test_branch_a_baseline -v`
- Training dry-run (2 batches): add `--max-batches 2` flag or equivalent guard before a full run
- Checkpoint integrity: load the saved `.pt` and confirm the metric in `benchmark_summary.json` matches the training log
- Early-stop integrity: confirm any truncated run reports `stopped_early=true` and includes a `stop_reason`
- Confusion-matrix integrity: run `python -m training.eval_branch_a --config config/config.yaml --run-name branch_a_test_eval` and confirm both JSON and Markdown reports are written under `runs/`

**Definition of done (default)**

- Unit tests pass for all affected modules.
- Output shape assertion exists for every new branch and committed interface: Branch B summary → `(B, 8)`, current Branch B module output → `(B, 32)`, Branch C → `(B, 28)`, Phase 3 logits → `(B,)`.
- No frozen branch weights change after an optimizer step — verified by test.
- Checkpoint saved with epoch, `model_state_dict`, `optimizer_state_dict`, and best metric.
- `runs/` log updated with the training run for the phase.
- **Phase gate metrics logged against forensics val set** (not CelebA proxy split).
- Progress snapshot updated when a gate flips.

**Plan hygiene**

- When a **gate** is met, update the **Progress snapshot** and the phase **Done when** if reality diverged from the original wording.
- **End of each week:** note what shipped and what slipped — one short list in the progress snapshot is enough.

---

## Week 1 — Setup + Branch A `[COMPLETE — Branch A diagnostic caveat remains]`

### Dev 1 — Model & Training

**Goal:** Branch A trains end-to-end. Checkpoint saved and gate-cleared. Branch A can be re-run on forensics only as a pair-labelled diagnostic, not as a canonical GAN single-image benchmark.

- [x] Project scaffold, `config.yaml`, `requirements.txt`
- [x] Experiment tracking setup (TensorBoard or W&B)
- [x] Implement `BranchA_CNN` (`models/branch_a.py`) — 5 conv blocks, SpectralNorm + BN, LeakyReLU(0.2) throughout, 2048-D flatten output
- [x] Implement `DiscriminatorPhase1` (`models/discriminator.py`) — Branch A + fusion FC head (2048 → 512 → 128 → 1)
- [x] Core training loop (`training/trainer.py`), BCE loss
- [x] Unit tests: forward-pass output shape `(B, 1)`, no NaN activations
- [x] Train Branch A on CelebA — proxy val balanced acc `1.0000`, F1 `1.0000` @ epoch `34` (noise-duplicate fakes; not a valid deepfake benchmark)
- [x] Save `checkpoints/phase1_branch_a_best.pt` + `runs/branch_a_baseline/benchmark_summary.json`
- [ ] **Re-evaluate Branch A on forensics val set** — pair-labelled diagnostic only; do not use as the deployment metric

### Dev 2 — Data & Eval

**Goal:** Dataset validated, flow cache complete, eval module interface defined. Forensics dataset downloaded and loader implemented.

- [x] Validate CelebA: 202,599 images, 178×218 native resolution
- [x] Implement `CelebAFramePairDataset` with adjacent-index fallback pairing (`data/celeba_loader.py`)
- [x] Augmentation pipeline: random horizontal flip, ColorJitter (brightness ±0.1, contrast ±0.1, saturation ±0.05), normalize to [-1, 1]
- [x] Data loader unit tests: shape checks, label balance, no NaN
- [x] Launch Farnebäck flow pre-computation (`data/precompute_flow.py`)
- [x] Flow cache verified: 202,599 files, 0 missing, 0 extra, shape `(2, 64, 64)` float32, ~7.0 GB
- [x] `tests.test_data.DataPipelineTestCase.test_flow_precompute_smoke` passing
- [x] Eval module skeleton (`evaluation/eval.py`) — `compute_balanced_accuracy`, `compute_f1`, `compute_auc_roc` stubs defined
- [x] **Stage forensics dataset** under `data/forensics/Data Set {1..4}/Data Set {1..4}/train|validation|test/real|fake`
- [x] **Implement forensics loader** (`data/forensics_loader.py`) — nested-root resolution, balanced limit, degenerate/adjacent pairing, aligned-root support
- [x] **Unit tests for forensics loader:** shape, label balance, no NaN, split normalization, nested root, aligned-root behavior

**Done when:** flow cache contains exactly 202,599 files; forensics loader is implemented and tested; Branch A is documented as non-canonical for single-image GAN scoring.

**Proxy result (CelebA):** acc `1.0000`, F1 `1.0000` — proxy gate cleared but not a valid benchmark.  
**Forensics val result:** pair diagnostic still pending; not a deployment metric.

---

## Week 2 — Branches B & C `[COMPLETE — forensics gates failed]`

Dev 1 owns Branch B. Dev 2 owns Branch C. Both run in parallel, but the remaining architectural question is whether Branch B's implemented 32-D learned expansion is a temporary Phase 2 convenience or the intended long-term contract.

### Dev 1 — Branch B (Spatiotemporal)

**Goal:** Branch B implemented and trained with Branch A frozen. `phase2_a_b.pt` saved. Evaluated on forensics val.

- [x] Implement `BranchB_Spatiotemporal` (`models/branch_b.py`)
  - Shared embed CNN: `frame_t`, `frame_t1` → 64-D each (tied weights, independent forward passes); LeakyReLU(0.2)
  - `velocity = e_t1 − e_t` (64-D)
  - `curvature = velocity / ‖velocity‖` (64-D, L2-normalized)
  - `acceleration` ≈ second-order approximation (64-D)
  - Aggregate `(mean, std, max)` over each of the three quantities → proposal-level **8-D summary**
  - Current implementation expands that summary through a small learned head to a **32-D output**
- [x] Implement `DiscriminatorPhase2` (`models/discriminator.py`)
  - Load `phase1_branch_a_best.pt`; set Branch A conv `requires_grad = False`
  - Current implementation concat `[branch_a_2048, branch_b_32]` → 2080-D into fusion head (2080 → 512 → 128 → 1)
  - Proposal target for the eventual three-branch model remains `[branch_a_2048, branch_b_8, branch_c_28]` → 2084-D
- [x] Write Phase 2 training script (`training/phase2_train.py`)
  - Optimizer: Adam (β₁=0.5, β₂=0.999), LR = 2e-4; only Branch B + fusion head params
  - Scheduler: CosineAnnealingLR; 20 epochs, batch size 64; loss: BCE
- [x] Unit tests (`tests/test_model.py`)
  - Branch B output shape `(B, 8)` ✓
  - Full Phase 2 forward pass output `(B, 1)` ✓
  - Branch A weights unchanged after optimizer step ✓
- [x] Train Branch B; save `checkpoints/phase2_a_b.pt`
- [ ] **Evaluate `phase2_a_b.pt` on forensics val set**; log score to `runs/phase2_a_b/benchmark_summary.json`

**Gate:** forensics-val balanced acc ≥ 88%, F1 ≥ 0.88; Branch A freeze verified by test.  
**Proxy result:** acc `1.0000`, F1 `1.0000` — proxy gate cleared but not a valid benchmark.  
**Forensics val result:** direct Phase 2 forensics neural score remains a tooling gap; do not report the proxy score as a forensics benchmark.

---

### Dev 2 — Branch C (Physics Dynamics)

**Goal:** Branch C implemented and trained with Branch A + B frozen. `phase3_a_b_c.pt` saved. Evaluated on forensics val.

> **Cache contract:** do not rename or regenerate `*_flow.pt` files during this week unless `identity_CelebA.txt` is explicitly introduced and cache regeneration is intentional. Keep Branch C on adjacent-index pairing to match the existing cache.

- [x] Update `CelebAFramePairDataset.__getitem__` to return `(frame_t, frame_t1, flow_tensor, label)`
  - Load `{frame_a_stem}_flow.pt` from `data/flow_cache/`
  - Unit test: returned flow tensor shape `(2, 64, 64)` ✓, no NaN ✓
- [x] Implement `BranchC_Physics` (`models/branch_c.py`)
  - **Optical flow features (20-D):** load cached dx/dy tensor; compute divergence, curl, gradient magnitude per pixel; aggregate `(mean, std, max, min, range)` over each → 15-D; global stats (mean magnitude, max magnitude, dominant direction histogram bins) → 5-D
  - **HSV photometrics (8-D):** convert `frame_t` and `frame_t1` from [-1,1] to [0,1] → RGB → HSV; per frame: `(mean_H, std_H, mean_S, mean_V)` → 4-D × 2 frames = 8-D
  - **Total: 28-D output**
- [x] Implement `DiscriminatorPhase3` (`models/discriminator.py`)
  - Load `phase2_a_b.pt`; freeze Branch A + Branch B (`requires_grad = False`)
  - Active contract: concat `[branch_a_2048, branch_b_32, branch_c_28]` → **2108-D**
- [x] Implement Hinge loss (`training/losses.py`)
- [x] Write Phase 3 training script (`training/phase3_train.py`)
- [x] Implement checkpoint save/resume (`training/checkpointing.py`, `training/phase3_trainer.py`)
- [x] Finalize eval module (`evaluation/eval.py`) — replace stubs; add `plot_confusion_matrix`
- [x] Unit tests — Branch C output shape `(B, 28)` ✓; Phase 3 forward pass `(B, 1)` ✓; A+B freeze verified ✓
- [x] Train Branch C; save `checkpoints/phase3_a_b_c.pt` — best val epoch `8`: balanced acc `0.8741`, F1 `0.9067`, AUC-ROC `0.9484`, loss `0.2726`
- [ ] **Launch forensics flow pre-computation** → `data/forensics_flow_cache/`
- [x] **Evaluate `phase3_a_b_c.pt` on forensics val set**; validation artifacts are in `runs/forensics_threshold/` and `runs/ensemble_ablation_forensics_val/`

**Gate:** forensics-val balanced acc ≥ 83%, F1 ≥ 0.80; A+B freeze verified; CelebA flow cache still 202,599 files; forensics flow cache started.  
**CelebA proxy result:** balanced acc `0.8741`, F1 `0.9067` — proxy gate cleared.  
**Forensics val result:** Phase 3 neural validation best balanced accuracy `0.5178` at threshold `0.06`; gate not cleared.

---

### Week 2 Critical Sync Point

> **End of Week 2** — proxy gates cleared. `phase2_a_b.pt` and `phase3_a_b_c.pt` both exist. Forensics validation shows the Phase 3 neural checkpoint does not transfer; Phase 2 direct forensics scoring still needs a dedicated wrapper if it must be reported separately.

---

## Week 3 — Full Ensemble Fine-tune

### Dev 1 — End-to-End Fine-tune

**Goal:** Fine-tune the active `2108-D` fusion contract with staged unfreezing. `phase4_ensemble.pt` saved. Ensemble experiments re-run on forensics val.

- [x] Implement `DiscriminatorPhase4` (`models/discriminator.py`) — staged unfreezing on `2108-D` contract
- [x] Implement asymmetric combined loss (`training/losses.py`)
- [x] Write Phase 4 fine-tune script (`training/phase4_finetune.py`) — 30 epochs staged: 10 fusion-only → 10 B+C → 10 Branch A tail
- [x] Run Phase 4 training; save `checkpoints/phase4_ensemble.pt` — balanced acc `0.8850`, F1 `0.8955`, AUC-ROC `0.9499`, TNR `0.72`, TPR `0.97`; Phase 3 preferred for balanced deployment
- [x] Implement all 7 ensemble combination experiments
- [x] Run full 7-combo ensemble job on proxy test split; record in `runs/ensemble_ablation/`
- [x] Prepare inference handoff artifact — `runs/<run>/inference_contract.json`
- [x] **Re-run 7-combo ensemble experiments on forensics val split**; results in `runs/ensemble_ablation_forensics_val/summary.json`. B+C RF pooled balanced accuracy `0.5000`, F1 `0.0000`, AUC-ROC `0.5505`; gate not cleared.

**Ensemble experiment matrix:**

| #     | Branches  | Classifier           |
| ----- | --------- | -------------------- |
| 1     | A only    | Logistic on logit    |
| 2     | B only    | Logistic on logit    |
| 3     | C only    | Logistic on logit    |
| 4     | A + B     | Random Forest        |
| 5     | A + C     | Random Forest        |
| **6** | **B + C** | **Random Forest** ⭐ |
| 7     | A + B + C | Random Forest        |

**Gate:** B+C ensemble forensics-val balanced acc ≥ 94.4%, F1 ≥ 0.93. Proxy-task gate not cleared. Phase 3 is the neural checkpoint baseline for threshold sweeps.

---

### Dev 2 — RF Ensemble + Ablation

**Goal:** RF classifiers trained for all 7 configs. Per-branch ablation and confusion matrix output complete. Forensics val used as the canonical eval split.

- [x] Implement `evaluation/ensemble.py` — feature extraction, RF training, evaluation
- [x] Implement `evaluation/threshold_sweep.py` — Phase 3 operating-point sweep; best threshold `0.61`
- [x] Implement `scripts/run_ensemble_ablation.py` — extract, run 7 configs, write confusion matrices, run threshold sweep, save `summary.json` / `summary.md`
- [x] Run RF ensemble for all 7 branch combinations on proxy held-out test split
- [x] Run threshold sweep on Phase 3 checkpoint — threshold `0.61`, balanced acc `0.8850`, F1 `0.8808`, TPR `0.8501`, TNR `0.9198`
- [x] Per-branch probe infrastructure: single-branch logistic probes + neural full-model logit view
- [x] Save confusion matrices for all 7 configs to `runs/ensemble_ablation/`
- [x] **Re-run RF ensemble and threshold sweep on forensics val split** — validation artifacts exist at `runs/ensemble_ablation_forensics_val/` and `runs/forensics_threshold/`

**Proxy result:** B+C RF balanced acc `0.8869`, F1 `0.8837`, AUC-ROC `0.9440` — gate not cleared.  
**Forensics val result:** B+C RF pooled balanced accuracy `0.5000`, F1 `0.0000`; gate not cleared.

---

## Week 4 — Eval & Hardening

### Dev 1 — Architecture Review + Experiment Support

- [x] Finalize all 7 ensemble results table; B+C is not deployment-ready on the completed OOD protocol
- [x] Architecture review: no orphaned branches, no unbounded tensor ops, no missing gradient guards
- [x] Support OOD eval — load Phase 3 as the neural baseline; accept image dir, output per-image scores
- [x] **Run final forensics test-set evaluation** — canonical locked benchmark in `runs/forensics_eval_final/summary.json`; gate not cleared

### Dev 2 — OOD Eval, Profiling, Report

**OOD evaluation (`evaluation/ood_eval.py`):**

- [x] Assemble local forensics OOD test sets under `data/forensics`
- [x] Implement OOD evaluation path via `evaluation/ood_eval.py` and `scripts/run_forensics_eval.py`
- [x] Run B+C ensemble on all local forensics datasets
- [x] Run Branch A baseline on the same local forensics datasets
- [x] **Run final held-out forensics test-set evaluation** — recorded in `runs/forensics_eval_final/summary.json`
- [x] **Download/stage and add forensics loader** (`data/forensics_loader.py`)

**Inference profiling:**

- [x] Profile forward pass latency per branch: CPU profile written to `runs/inference_profile/summary.json`; MPS/CUDA unavailable to this profiler run
- [ ] If flow pre-compute is a bottleneck: parallelize with `multiprocessing.Pool`; evaluate CUDA Farnebäck if GPU is available

**Final eval report:**

- [x] Consolidated results table — all 7 transfer ensemble configs × forensics validation/test OOD
- [x] Per-branch ablation table
- [x] Confusion matrices (per OOD dataset + pooled); full in-domain matrices remain pending
- [x] **Forensics test-set final scores** (canonical benchmark)
- [x] Inference time profile
- [x] Deployment recommendation: do not ship B+C or A+B+C from this training run; OOD gate failed

**Done when:** forensics test-set eval complete; final report updated with canonical benchmark scores; inference profile logged to `runs/`; `docs/final-report.md` committed.

---

## Architecture Reference

### Branch Dimensions

| Branch                            | Dim                         | Signal                                                               |
| --------------------------------- | --------------------------- | -------------------------------------------------------------------- |
| A — CNN Spatial                   | 2048-D                      | Static texture & structure (5 conv blocks, SpectralNorm + BN)        |
| B — Spatiotemporal                | 8-D summary → 32-D expanded | Shared-encoder temporal stats: velocity + cosine/L2/sign consistency |
| C — Physics Dynamics              | 28-D                        | Optical flow div/curl/grad (20-D) + HSV photometrics (8-D)           |
| **Concatenated (active runtime)** | **2108-D**                  | Fusion FC input                                                      |

### Hyperparameters

| Parameter        | Phases 1–3              | Phase 4                                                                        |
| ---------------- | ----------------------- | ------------------------------------------------------------------------------ |
| Optimizer        | Adam (β₁=0.5, β₂=0.999) | same                                                                           |
| Learning rate    | 2e-4                    | staged: 5e-5 → 2e-5 → 5e-6                                                     |
| Batch size       | 64                      | 64                                                                             |
| Epochs           | 20                      | 30 total: 10 fusion-only, 10 B+C, 10 Branch A tail                             |
| Scheduler        | CosineAnnealingLR       | same                                                                           |
| Loss             | BCE                     | AsymmetricCombinedLoss: 0.7 × fake-positive BCE + 0.3 × hinge, real_weight=1.5 |
| Dropout (fusion) | 0.3                     | 0.3                                                                            |

---

## Checkpoint Registry

| File                      | Phase | Contents                                            | Status                                                                                                                                          |
| ------------------------- | ----- | --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `phase1_branch_a_best.pt` | 1     | Branch A conv + FC                                  | Proxy gate cleared; Branch A forensics pair diagnostic remains non-canonical for GAN single-image scoring                                        |
| `phase2_a_b.pt`           | 2     | Branch A (frozen) + Branch B + FC                   | Proxy gate cleared; direct Phase 2 forensics neural scoring is still a tooling gap because `ood_eval` is Phase 3/4 feature based                |
| `phase3_a_b_c.pt`         | 3     | A + B (frozen) + Branch C + FC                      | CelebA proxy: balanced acc `0.8741`, F1 `0.9067`, AUC-ROC `0.9484`; forensics val/test gates fail; current baseline for recovery experiments   |
| `phase4_ensemble.pt`      | 4     | Staged fine-tune: fusion-only → B+C → Branch A tail | Characterized: balanced acc `0.8850`, F1 `0.8955`, AUC-ROC `0.9499`, TNR `0.72`, TPR `0.97`; not preferred over Phase 3 for balanced deployment |

> **Retain all four checkpoints** until forensics val/test evaluation is complete. Do not discard Phase 4 or earlier checkpoints before the forensics test-set scores are in — the preferred checkpoint may change based on real deepfake benchmark results.

---

## Expected Final Results (Proposal Targets)

| Configuration | Auth %    | Synth %   | F1       | Notes                           |
| ------------- | --------- | --------- | -------- | ------------------------------- |
| Branch A only | 77.8%     | 77.8%     | 0.70     | Phase 1 gate                    |
| Branch B only | 88.9%     | 94.4%     | 0.91     |                                 |
| Branch C only | 83.3%     | 83.3%     | 0.80     |                                 |
| A + B         | 89.5%     | 89.5%     | 0.88     |                                 |
| A + C         | 88.9%     | 88.9%     | 0.85     |                                 |
| **B + C**     | **94.4%** | **94.4%** | **0.93** | ⭐ Proposal target              |
| A + B + C     | 89.5%     | 89.5%     | 0.86     | Branch A dilutes OOD robustness |

> All targets above are evaluated against the **forensics val/test set**, not the CelebA proxy split. Current proxy-task B+C RF result: balanced acc `0.8869`; locked forensics test B+C RF result: balanced acc `0.5000` — both below the proposal gate. The `runs/forensics_eval_final/` run is the canonical measurement.

---

## Risk Register

| Risk                                                                             | Likelihood | Impact | Mitigation                                                                                                                 |
| -------------------------------------------------------------------------------- | ---------- | ------ | -------------------------------------------------------------------------------------------------------------------------- |
| Branch A dominates gradients in Phase 4                                          | High       | High   | Phased freeze ensures independent feature learning; gradient scaling if Phase 4 still shows Branch A dominance             |
| `identity_CelebA.txt` introduced mid-cache                                       | Medium     | High   | Keep Branch C on adjacent-index pairing OR regenerate cache before Phase 3 training; never silently mix pairing strategies |
| Shared Branch B/Branch A encoder tail overfits or drifts BN stats during Phase 2 | Medium     | Medium | Freeze blocks 0-2, keep blocks 3-4 in train mode only, enforce partial-freeze behavior in unit tests                       |
| Flow cache corrupted or stems mismatched                                         | Low        | High   | `test_flow_precompute_smoke` must pass before Phase 3 train; verify file count after every run                             |
| Forensics flow cache pairing mismatch                                            | Medium     | High   | Branch C must use consistent adjacent-index pairing for forensics cache, or regenerate both caches together                |
| Forensics dataset not available in time                                          | Low        | High   | Download immediately; it is the canonical eval dataset — no team-lead handoff is valid without it                          |
| Phase gate metrics reported from proxy split instead of forensics val            | High       | High   | All `benchmark_summary.json` files must be updated with forensics val scores before handoff                                |
| OOD test sets unavailable in Week 4                                              | Medium     | Medium | Source and stage OOD data during Week 3 in parallel with ensemble training                                                 |
| Overfitting to CelebA in Phase 4                                                 | Medium     | High   | Forensics val/test eval mandatory before Phase 4 sign-off                                                                  |

---

## Standing Rules

- **Freeze before you train.** No phase training begins without a passing unit test confirming prior branch weights are frozen.
- **Cache contract is inviolable.** `{frame_a_stem}_flow.pt`, shape `(2, 64, 64)`, adjacent-index partner rule. Any deviation is an explicit decision requiring cache regeneration.
- **B+C is only the proposal deployment config.** Do not ship it unless a real OOD run clears the gate.
- **OOD eval is not optional.** Week 4 is not done until forensics test-set numbers exist and the result is recorded honestly, pass or fail.
- **Checkpoints are the handoff artifact.** Each week ends with a saved checkpoint. If the gate is not cleared, the checkpoint is still saved and the miss is noted in the progress snapshot.
- **One training script per phase.** Do not fold phases into one script; each script is its own audit trail.
- **Forensics is the eval dataset.** CelebA val/test is for training monitoring only. All metrics reported to the team lead must come from the forensics val or test split.

---

## Weekly rhythm

| Day           | Activity                                                                                                                       |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Monday**    | Review prior week gate; pick concrete tasks for the week; confirm prior checkpoint loads cleanly before writing new code       |
| **Wednesday** | Mid-week check — if a branch is not converging, decide to adjust LR or descope; do not let one failing branch block the other  |
| **Thursday**  | Integrate risky pieces (freeze tests, data loader contract changes) so Friday is not the first time they run together          |
| **Friday**    | Full training dry-run or checkpoint validation on physical hardware; update **Progress snapshot**; note what shipped / slipped |
