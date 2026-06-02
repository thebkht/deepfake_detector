# Hybrid Three-Branch GAN Discriminator — Master Plan

> Deepfake Face Detection · Based on Barrington & Farid, CVPR Workshop 2026  
> Training: CelebA (202,599 images) · `jessicali9530/celeba-dataset`  
> Val / Test: Real & Fake Images Dataset for Image Forensics · `shivamardeshna/real-and-fake-images-dataset-for-image-forensics`
>
> Doc basis: refreshed on 2026-05-29 from the project proposal and current repository state

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Build Plan](#3-build-plan)
4. [Team & Task Split](#4-team--task-split)
5. [File & Module Structure](#5-file--module-structure)
6. [Data Pipeline](#6-data-pipeline)
7. [Training Strategy (4 Phases)](#7-training-strategy-4-phases)
8. [Loss Functions](#8-loss-functions)
9. [Evaluation & Metrics](#9-evaluation--metrics)
10. [Expected Performance Targets](#10-expected-performance-targets)
11. [Risks & Mitigations](#11-risks--mitigations)
12. [Dependencies](#12-dependencies)

---

## 1. Project Overview

### Problem

Standard single-branch GAN discriminators achieve 95%+ accuracy on in-distribution content but **collapse to ~52% accuracy** when encountering novel generation methods (style transfer, reenactment, diffusion-based synthesis). The goal is to recover accuracy above **94% on out-of-domain content**.

### Solution

A **hybrid three-branch discriminator** that captures orthogonal signals:

| Branch               | Signal Type                            | Why It Helps                                                     |
| -------------------- | -------------------------------------- | ---------------------------------------------------------------- |
| A — CNN Spatial      | Static frame-level texture & structure | Strong in-domain baseline                                        |
| B — Spatiotemporal   | Temporal embedding velocity/curvature  | Catches lip-sync deepfakes, expression swaps                     |
| C — Physics Dynamics | Optical flow + HSV photometrics        | Catches skin flicker, implausible flow, color temperature shifts |

**Key result from proposal:** B + C ensemble → **94.4% balanced accuracy, F1 = 0.93** (vs. 52% baseline on OOD content).

### Repository reality check

The proposal is the target design, not the current implementation state.

- Implemented now: Branch A baseline, Branch B temporal summary branch, Branch C physics branch, Phase 2 and Phase 3 training paths, CelebA pair loader, offline flow precompute, checkpoint helpers, and evaluation metrics/plots
- Implemented now: Phase 4 fine-tuning path, Phase 4 checkpoint metadata, staged unfreezing, fake-positive asymmetric BCE+hinge loss, and inference handoff artifact wiring
- Current checkpoint choice: Phase 3 is the best deployment-style candidate under the balanced objective; the final Phase 4 run improved balanced accuracy/AUC only slightly but lowered F1 and real-class TNR
- Implemented and run on the proxy test split: random-forest ensemble tooling, single-branch probe/ablation reporting, and Phase 3 threshold-sweep tooling
- Current ensemble result: B+C RF did not clear the proposal gate, reaching balanced accuracy `0.8869`, F1 `0.8837`, and AUC-ROC `0.9440`; A+B+C RF was strongest on the proxy task at balanced accuracy `0.8992`, F1 `0.8962`, and AUC-ROC `0.9471`
- Implemented and run on local forensics OOD data: full transfer evaluation under `runs/forensics_eval/` across `20,905` images; B+C RF fails with pooled balanced accuracy `0.4716`, F1 `0.4981`, and AUC-ROC `0.4572`
- Important delta: the active runtime contract remains `2048 + 32 + 28 = 2108`; proposal-parity `2084-D` fusion is not the current load-compatible path

> ⚠️ **Branch A benchmark caveat:** The recorded Branch A result (1.0 balanced accuracy, 1.0 F1) was produced against Gaussian noise-duplicate fake pairs, not real deepfake images. This is a proxy-task smoke test, not a meaningful deepfake detection score. Branch A must be re-evaluated against the forensics val set before any results are reported to the team lead.

---

## 2. Architecture

### 2.1 High-Level Diagram

```mermaid
flowchart LR
  subgraph branchA ["Branch A — CNN Spatial"]
    FA["frame_t"] --> ENC_A["BranchAEncoder\nno_grad"]
    ENC_A --> featA["2048-D"]
  end

  subgraph branchB ["Branch B — Spatiotemporal"]
    FA --> ENC_S["BranchAEncoder\nshared"]
    FT1["frame_t1"] --> ENC_S
    ENC_S --> e_t["e_t 2048-D"]
    ENC_S --> e_t1["e_t1 2048-D"]
    e_t --> SUM["8-D temporal summary\nvel · cos_sim · l2_dist · sign_consistency"]
    e_t1 --> SUM
    SUM --> EXP["expander 8→32\nLayerNorm + Linear + LeakyReLU"]
  end

  subgraph branchC ["Branch C — Physics Dynamics"]
    FA --> FLOW["Farnebäck optical flow\n+ HSV photometrics"]
    FT1 --> FLOW
    FLOW --> featC["28-D\nflow div/curl/grad + HSV stats"]
  end

  featA --> FUS["Fusion FC Head\n2108-D → 512 → 128 → 1"]
  EXP --> FUS
  featC --> FUS
  FUS --> OUT(["Real / Fake"])
```

### 2.2 Branch A — CNN Spatial Features

Operates on a **single 64×64×3 frame**. Five convolutional blocks with SpectralNorm + BatchNorm.

| Layer   | Operation                      | Output Shape | Norm                           |
| ------- | ------------------------------ | ------------ | ------------------------------ |
| Input   | —                              | 64×64×3      | —                              |
| Conv 1  | Conv2d(3→64, k=4, s=2, p=1)    | 32×32×64     | None (no BN on first layer)    |
| Conv 2  | Conv2d(64→128, k=4, s=2, p=1)  | 16×16×128    | SpectralNorm + BN              |
| Conv 3  | Conv2d(128→256, k=4, s=2, p=1) | 8×8×256      | SpectralNorm + BN              |
| Conv 4  | Conv2d(256→512, k=4, s=2, p=1) | 4×4×512      | SpectralNorm + BN              |
| Conv 5  | Conv2d(512→512, k=4, s=2, p=1) | 2×2×512      | SpectralNorm (no BN before FC) |
| Flatten | 2×2×512 → vector               | **2048-D**   | —                              |

Activation: **LeakyReLU(0.2)** throughout.

### 2.3 Branch B — Spatiotemporal Embedding Derivatives

Operates on a **consecutive frame pair**. The proposal describes a compact embedding CNN, while the current implementation reuses the pretrained `BranchAEncoder` and computes summary statistics in that shared feature space.

```mermaid
flowchart LR
  frame_t["frame_t"] --> CNN1["Shared BranchAEncoder"]
  frame_t1["frame_t1"] --> CNN2["Shared BranchAEncoder"]
  CNN1 --> e_t["e_t 2048-D"]
  CNN2 --> e_t1["e_t1 2048-D"]
  e_t --> VEL["velocity = e_t1 − e_t"]
  e_t1 --> VEL
  VEL --> CUR["curvature = velocity / ‖velocity‖"]
  VEL --> ACC["accel ≈ second-order approx"]
  CUR --> STATS["Summary stats\nmean, std, max × 3 quantities"]
  ACC --> STATS
  VEL --> STATS
  STATS --> OUT["8-D output"]
  OUT --> EXP["expander 8 to 32-D"]
```

**What it catches:** Lip-sync deepfakes, expression swaps, discontinuous feature trajectories.

### 2.4 Branch C — Physics-Based Dynamics

Accepts either **pre-computed features** (offline, recommended for speed) or **raw frame pairs**.

```
Optical flow (div / curl / grad)     → 20-D
HSV photometrics (h, s, v mean/std)  →  8-D
─────────────────────────────────────────────
Total                                → 28-D
```

**What it catches:** Skin tone flicker, implausible optical flow patterns, color temperature inconsistencies.

> **Note:** For production, pre-compute optical flow offline using Farnebäck (OpenCV) and cache as `.pt` tensors alongside each image. This avoids flow computation bottleneck during training.

### 2.5 Fusion Head

| Contract | Layer    | In → Out   | Activation                       |
| -------- | -------- | ---------- | -------------------------------- |
| Proposal | Linear 1 | 2084 → 512 | LeakyReLU(0.2) + Dropout(0.3)    |
| Current  | Linear 1 | 2108 → 512 | LeakyReLU(0.2) + Dropout(0.3)    |
| Both     | Linear 2 | 512 → 128  | LeakyReLU(0.2)                   |
| Both     | Linear 3 | 128 → 1    | — (logits; sigmoid at inference) |

### 2.6 Phase-by-phase tensor contracts

This section keeps the proposal contract and the current code path separate.

| Stage                           | Tensor contract                             | Status                                 |
| ------------------------------- | ------------------------------------------- | -------------------------------------- |
| Proposal Branch A encoder       | `2048-D` per frame                          | Implemented                            |
| Proposal Branch B summary       | `8-D` per frame pair                        | Implemented as an intermediate summary |
| Current Phase 2 Branch B output | `32-D` learned expansion of the 8-D summary | Implemented                            |
| Proposal Branch C output        | `28-D` per frame pair                       | Implemented                            |
| Proposal final fusion           | `2048 + 8 + 28 = 2084-D`                    | Not implemented                        |
| Current Phase 2 fusion          | `2048 + 32 = 2080-D`                        | Implemented                            |
| Current Phase 3 fusion          | `2048 + 32 + 28 = 2108-D`                   | Implemented                            |

---

## 3. Build Plan

### Project Checkpoint — 2026-05-26

Current status from the repository state:

- **Week 1 is substantially complete.** The local CelebA tree is present at `data/celeba/img_align_celeba` with **202,599 images**, the Week 1 data pipeline and tests exist, and the Branch A baseline has already produced `checkpoints/phase1_branch_a_best.pt`.
- **Branch A checkpoint is real and measurable.** `runs/branch_a_baseline/benchmark_summary.json` reports best validation metrics of **1.0000 balanced accuracy** and **1.0000 F1** at epoch **34**, which clears the Week 1 gate — but see the caveat in §1: these metrics are against noise-duplicate fakes and must be re-run against the forensics val set before team-lead handoff.
- **Farnebäck cache is already complete on this machine.** Verified on **2026-05-18**: `data/flow_cache` contains **202,599** `*_flow.pt` files with **0 missing / 0 extra** stems against `discover_celeba_images`, sample tensors have shape `(2, 64, 64)` and `float32` dtype, the cache occupies about **7.0 GB**, and `tests.test_data.DataPipelineTestCase.test_flow_precompute_smoke` passes.
- **Week 2 Branch C must preserve the cache contract.** Cached flow filenames are `{frame_a_stem}_flow.pt`, and each tensor is computed against the adjacent-index partner rule used by `data/precompute_flow.py`. The loader now uses cross-identity proxy negatives when identity labels are available, so Branch C must either stay on explicit adjacent-index pairing or use a regenerated cache that matches the new pair selection rule before training.
- **Week 2 Dev 1 code is now in place.** `models/branch_b.py`, `models/discriminator.py`, `training/phase2_trainer.py`, `training/phase2_train.py`, and `tests/test_model.py` now exist. Branch B's proposal-level 8-D layout and acceleration proxy are locked by a golden regression test, and the current implementation expands that summary to a learned 32-D feature before Phase 2 fusion. Phase 2 also includes a real Branch A freeze test plus Phase 1 encoder load/remap coverage.
- **Current training runs now use guarded stopping.** Branch A and Phase 2 trainers stop early when validation loss shows sustained overfitting, and each phase also has a branch-specific validation-loss ceiling after warmup.
- **Phase 2 is gate-cleared.** `checkpoints/phase2_a_b.pt` now exists with `phase == 2`; the saved checkpoint reports best validation metrics of **1.0000 balanced accuracy** and **1.0000 F1** at epoch **2**, and `runs/phase2_a_b/benchmark_summary.json` matches those values.
- **Week 2 Dev 2 training is now complete.** `checkpoints/phase3_a_b_c.pt` exists and matches `runs/phase3_a_b_c_w2/benchmark_summary.json`. The best validation result occurs at epoch **8** with **0.8741 balanced accuracy**, **0.9067 F1**, **0.9484 AUC-ROC**, and **0.2726 loss**, which clears the configured Phase 3 gate.
- **Week 3 Phase 4 has been executed.** Phase 4 uses the locked `2108-D` contract with stage-aware early stopping across a 30-epoch plan: 10 fusion-only epochs, 10 Branch B+C epochs, then 10 Branch A-tail epochs. Final comparison: Phase 3 `0.8790` balanced accuracy / `0.9072` F1 / `0.9480` AUC-ROC / `0.78` TNR / `0.94` TPR; Phase 4 `0.8850` balanced accuracy / `0.8955` F1 / `0.9499` AUC-ROC / `0.72` TNR / `0.97` TPR. The asymmetric fine-tune increased fake recall but worsened real specificity, so Phase 3 remains the preferred neural checkpoint.
- **Week 3 ensemble run is complete on the proxy test split.** `runs/ensemble_ablation/summary.md` reports `13,074` balanced test examples, with B+C RF at **0.8869 balanced accuracy / 0.8837 F1 / 0.9440 AUC-ROC**, below the proposal gate. A+B+C RF is the strongest proxy-task probe at **0.8992 balanced accuracy / 0.8962 F1 / 0.9471 AUC-ROC**. The neural Phase 3 threshold sweep selects threshold `0.61` for **0.8850 balanced accuracy**, F1 `0.8808`, TPR `0.8501`, and TNR `0.9198`.
- **Week 4 forensics OOD evaluation is complete.** `runs/forensics_eval/summary.json` is the Dev 2 handoff artifact. The pooled transfer results fail: B+C RF reaches **0.4716 balanced accuracy / 0.4981 F1 / 0.4572 AUC-ROC**, and `docs/final-report.md` records the negative transfer conclusion.

### Milestones

```mermaid
gantt
  title Project Timeline
  dateFormat YYYY-MM-DD
  section Week 1
    Setup + Branch A (Dev 1)          :done, 2026-05-15, 7d
    Data + Flow + Eval module (Dev 2) :done, 2026-05-15, 7d
  section Week 2
    Branch B (Dev 1)                  :done, 2026-05-22, 7d
    Flow cache + Branch C (Dev 2)     :done, 2026-05-22, 7d
  section Week 3
    Ensemble fine-tune (Dev 1)        :done, 2026-05-29, 3d
    RF ensemble + ablation (Dev 2)    :done, 2026-05-29, 4d
  section Week 4
    Ensemble experiments (Dev 1)      :2026-06-05, 7d
    OOD + profiling + report (Dev 2)  :2026-06-05, 7d
```

> **Compression rationale:** Setup and Branch A are merged into Week 1 by running data pipeline work (Dev 2) in parallel with scaffold + model work (Dev 1). Branches B and C are built in parallel in Week 2 since they are independent of each other. Eval is tightened to one week by preparing the eval harness during Week 3 alongside training.

### Week 1 — Setup + Branch A

- [x] CelebA already present locally — dataset validation only
- [x] Write `CelebAFramePairDataset` with identity-based pair sampling
- [x] Write unit tests for data loader: shape checks, label balance, no NaN
- [x] Farnebäck optical flow pre-computation complete (202,599 files, ~7 GB)
- [x] Set up experiment tracking (TensorBoard)
- [x] Write `config.yaml` with all hyperparameters
- [x] Implement `BranchA_CNN` + `DiscriminatorPhase1`
- [x] Train Branch A on CelebA proxy real/fake pairs; save `checkpoints/phase1_branch_a_best.pt`
- [ ] **Re-evaluate Branch A on forensics val set** — required before team-lead handoff (proxy 1.0 score is not a valid deepfake benchmark)
- [x] **Download forensics dataset** (`shivamardeshna/real-and-fake-images-dataset-for-image-forensics`)
- [ ] **Implement `ForensicsDataset` loader** — resize 256→64, balanced 50/50 real/fake, val/test split

### Week 2 — Branches B & C (parallel)

- [x] **Dev 1:** Implement `BranchB_Spatiotemporal` + `DiscriminatorPhase2` (Branch A frozen); smoke-verify load/freeze/tests and Phase 2 training path
- [x] **Dev 1:** Run full Phase 2 training; target ≥88% accuracy; save `checkpoints/phase2_a_b.pt`
- [x] **Dev 2:** Finalize flow cache `.pt` files; implement `BranchC_Physics`; implement Hinge loss
- [x] **Dev 2:** Run full Branch C / Phase 3 training with A+B frozen; target ≥83% accuracy; save `checkpoints/phase3_a_b_c.pt`
- [x] **Dev 2:** Build balanced accuracy / F1 / AUC-ROC eval module; checkpoint save/resume
- [ ] **Dev 2:** Pre-compute Farnebäck flow for forensics images → `data/forensics_flow_cache/`

### Week 3 — Full Ensemble Fine-tune

- [x] Dev 1: 30-epoch staged ensemble fine-tune with asymmetric BCE+hinge loss; save `phase4_ensemble.pt`
- [x] Dev 1: Characterize Phase 4 result; Phase 3 remains preferred balanced checkpoint
- [x] Dev 1/Dev 2: Run all 7 ensemble combination experiments; write summary
- [ ] Target: **B+C ensemble ≥ 94.4% balanced accuracy, F1 ≥ 0.93** — not cleared on proxy or forensics split
- [x] Dev 2: Implement confusion-matrix output and per-branch probe/ablation reporting
- [ ] **Dev 1/Dev 2: Re-run ensemble experiments using forensics val set** as the canonical eval split

### Week 4 — Eval & Hardening

- [x] Evaluate on local forensics OOD test set — negative transfer result recorded
- [x] Run threshold sweep on Phase 3 checkpoint
- [x] Run proxy-task ablation/probe pass: each branch independently, all pairs, full triple
- [ ] **Run final evaluation on forensics test set** (held-out 50% of forensics dataset) — canonical benchmark for team-lead report
- [ ] Profile inference time; optimize Branch C flow pre-computation
- [x] Write final eval report (`docs/final-report.md`)

---

## 4. Team & Task Split

Two developers, four weeks, split by model vs. data/eval ownership.

> **Merge rationale:** With two people, training and evaluation responsibilities (previously Dev 3) are absorbed into Dev 1 and Dev 2 respectively. Branch B and C are trained sequentially rather than in parallel — Dev 1 handles Branch B while Dev 2 handles Branch C in the same week, which is still achievable since they don't share code. Dev 2 absorbs all evaluation and reporting work previously owned by Dev 3.

### Roles

| Developer | Role                       | Primary Ownership                                                                             |
| --------- | -------------------------- | --------------------------------------------------------------------------------------------- |
| Dev 1     | Model & training           | `models/`, `training/` — all three branches, training scripts, ensemble fine-tune             |
| Dev 2     | Data, physics & evaluation | `data/`, `evaluation/` — CelebA loader, forensics loader, flow cache, Branch C, eval, reports |

### Per-Developer Task Breakdown

**Dev 1 — Model & training**

| Week | Tasks                                                                                                                                                                                                                                                     |
| ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | Project scaffold, `config.yaml`, requirements, experiment tracking setup; `BranchA_CNN` + `DiscriminatorPhase1`; core training loop, BCE loss; unit tests; train Branch A on CelebA; **re-evaluate on forensics val set**; save `phase1_branch_a_best.pt` |
| 2    | `BranchB_Spatiotemporal` + `DiscriminatorPhase2` (Branch A frozen); Phase 2 training script; full-gate Branch B training; evaluate on forensics val; `phase2_a_b.pt` ready for Dev 2                                                                      |
| 3    | 30-epoch staged ensemble fine-tune with asymmetric BCE+hinge loss; save and characterize `phase4_ensemble.pt`; keep Phase 3 as the balanced checkpoint baseline; re-run ensemble experiments on forensics val                                             |
| 4    | All 7 ensemble combination experiments on forensics test set; Phase 3 threshold sweep; architecture review; support final report                                                                                                                          |

**Dev 2 — Data, physics & evaluation**

| Week | Tasks                                                                                                                                                                                                                                                                                                                              |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | CelebA validation (data already present); `CelebAFramePairDataset` + augmentation pipeline + data loader unit tests; CelebA flow cache verified (202,599 files); **download forensics dataset**; implement `ForensicsDataset` loader (resize 256→64, 50/50 balanced, val/test split); balanced accuracy / F1 / AUC-ROC eval module |
| 2    | `BranchC_Physics`; Phase 3 training script (A+B frozen); train Branch C on CelebA; evaluate on forensics val; save `phase3_a_b_c.pt`; Hinge loss; **launch forensics flow cache pre-computation** → `data/forensics_flow_cache/`                                                                                                   |
| 3    | Random forest ensemble (B+C, A+B, A+C, A+B+C); confusion matrix + per-branch ablation module; checkpoint save/resume; re-run ensemble on forensics val                                                                                                                                                                             |
| 4    | **Final test-set evaluation on forensics dataset** (canonical team-lead benchmark); inference profiling; final eval report                                                                                                                                                                                                         |

### 4-Week Timeline

| Team  | Week 1                              | Week 2                    | Week 3                 | Week 4                                |
| ----- | ----------------------------------- | ------------------------- | ---------------------- | ------------------------------------- |
| Dev 1 | Scaffold + Branch A + ForensicsEval | Branch B + forensics val  | Ensemble fine-tune     | Ensemble experiments + forensics test |
| Dev 2 | Data + ForensicsLoader + eval mod   | Branch C + forensics flow | RF ensemble + ablation | Forensics test + report               |

### Critical Sync Points

```mermaid
sequenceDiagram
  participant D1 as Dev 1
  participant D2 as Dev 2
  Note over D1,D2: Mid Week 1
  D2->>D1: CelebA + ForensicsDataset loaders ready
  D2->>D1: eval module interface stable
  Note over D1,D2: End of Week 1
  D1->>D2: phase1_branch_a_best.pt + forensics val benchmark
  Note over D1,D2: End of Week 2
  D1->>D2: phase2_a_b.pt + forensics val score
  D2->>D1: phase3_a_b_c.pt + forensics flow cache ready
  Note over D1,D2: End of Week 3
  D1->>D2: phase4_ensemble.pt + Phase 3 baseline recommendation
  Note over D1,D2: End of Week 4
  D2->>D1: forensics test-set results + final report
```

---

## 5. File & Module Structure

```
deepfake_detector/
│
├── config/
│   └── config.yaml                  # All hyperparameters
│
├── data/
│   ├── celeba_loader.py             # CelebAFramePairDataset + DataLoader factory
│   ├── forensics_loader.py          # ForensicsDataset (val/test only, 256→64 resize)
│   ├── precompute_flow.py           # Farnebäck flow pre-computation script
│   └── augmentations.py             # Shared transforms
│
├── models/
│   ├── discriminator.py             # HybridDiscriminator + all branches
│   ├── branch_a.py                  # BranchA_CNN
│   ├── branch_b.py                  # BranchB_Spatiotemporal
│   └── branch_c.py                  # BranchC_Physics
│
├── training/
│   ├── trainer.py                   # Main training loop
│   ├── losses.py                    # BCE, hinge, and asymmetric Phase 4 loss implementations
│   ├── phase1_train.py              # Branch A only
│   ├── phase2_train.py              # A + B (frozen A)
│   ├── phase3_train.py              # A + B + C (frozen A, B)
│   └── phase4_finetune.py           # Staged Phase 4 fine-tune
│
├── evaluation/
│   ├── eval.py                      # Balanced accuracy, F1, confusion matrix
│   ├── ensemble.py                  # 7-combo logistic/RF ensemble probes
│   ├── inference_handoff.py         # Phase 4 → Week 4 inference contract artifact
│   └── threshold_sweep.py           # Phase 3 threshold operating-point sweep
│
├── checkpoints/                     # Saved model weights (gitignored)
│
├── runs/                            # TensorBoard logs (gitignored)
│
├── scripts/
│   ├── download_celeba.sh           # Kaggle API download helper — training data
│   ├── download_forensics.sh        # Kaggle API download helper — val/test data
│   ├── eval_pred_all_branches.py    # Prediction CSV/confusion export for checkpoints
│   └── run_ensemble_ablation.py     # 7-combo ensemble + threshold sweep runner
│
├── tests/
│   ├── test_model.py                # Shape/forward-pass unit tests
│   └── test_data.py                 # Data loader unit tests
│
├── requirements.txt
└── README.md
```

---

## 6. Data Pipeline

### Dataset Strategy (Team Lead Directive)

| Role                  | Dataset                                | Source                                                                    |
| --------------------- | -------------------------------------- | ------------------------------------------------------------------------- |
| **Training only**     | CelebA                                 | `kaggle: jessicali9530/celeba-dataset`                                    |
| **Validation & Test** | Real & Fake Images for Image Forensics | `kaggle: shivamardeshna/real-and-fake-images-dataset-for-image-forensics` |

The forensics dataset contains genuine real/fake image pairs (not noise duplicates), making it a meaningful out-of-distribution benchmark. CelebA provides the volume needed for training all three branches. The two datasets must **never be mixed during training** — forensics data is strictly held out for val/test.

> ⚠️ **Benchmark validity:** Because the forensics dataset contains actual AI-generated and manipulated faces, val/test metrics on it are a genuine measure of deepfake detection ability. The Branch A and Phase 2 proxy scores (1.0 balanced accuracy) were produced against Gaussian noise duplicates and are not valid deepfake benchmarks. All phase benchmarks reported to the team lead must use forensics val scores.

### Training Dataset — CelebA

| Property          | Value                        |
| ----------------- | ---------------------------- |
| Total images      | 202,599                      |
| Identities        | 10,177                       |
| Native resolution | 178×218                      |
| Target resolution | 64×64                        |
| Attributes        | 40 binary labels per image   |
| License           | Non-commercial research only |
| Role              | Training only (Phases 1–4)   |

#### CelebA Frame Pair Sampling

```
Real pair:   two images of the same celebrity identity
             (simulate consecutive frames of authentic video)

Legacy proxy: single image + small Gaussian noise duplicate
              used by earlier historical checkpoints only

Current proxy: anchor image + different-identity image when labels exist
               fallback to distant-index pairing without identity labels
               → replace with GAN/diffusion outputs during full deepfake training
```

#### CelebA Split (used only for training; not used for val/test reporting)

| Split | Image Range       | Count   |
| ----- | ----------------- | ------- |
| Train | 1 – 162,770       | 162,770 |
| Val   | 162,771 – 182,637 | 19,867  |
| Test  | 182,638 – 202,599 | 19,962  |

### Validation & Test Dataset — Real & Fake Images for Image Forensics

| Property          | Value                                                                     |
| ----------------- | ------------------------------------------------------------------------- |
| Source            | `kaggle: shivamardeshna/real-and-fake-images-dataset-for-image-forensics` |
| Content           | Mixed real photographs + AI-generated/manipulated fake images             |
| Native resolution | 256×256                                                                   |
| Target resolution | 64×64 (resize to match model input)                                       |
| Role              | Validation and test only — **never used for training**                    |

#### Forensics Dataset Split

Shuffle and split the forensics dataset at balanced 50/50 real/fake ratio as specified by team lead:

| Split | Fraction              | Purpose                                                            |
| ----- | --------------------- | ------------------------------------------------------------------ |
| Val   | 50% of forensics data | Hyperparameter selection, checkpoint ranking, per-phase benchmarks |
| Test  | 50% of forensics data | Final benchmark reported to team lead — held out until Week 4      |

#### Forensics Frame Pair Strategy

Since the forensics dataset contains single images (not video frames):

```
Real pair:   two different real images from the forensics real subset
Fake pair:   two different fake images from the forensics fake subset
```

### Augmentations

| Split              | Augmentations                                                                                                |
| ------------------ | ------------------------------------------------------------------------------------------------------------ |
| CelebA train       | Random horizontal flip, ColorJitter (brightness ±0.1, contrast ±0.1, saturation ±0.05), normalize to [-1, 1] |
| Forensics val/test | Resize to 64×64, normalize to [-1, 1] — **no augmentation**                                                  |

### Optical Flow Pre-computation

```bash
# CelebA flow cache (training) — already complete on local workspace
python data/precompute_flow.py \
  --img-dir data/celeba/img_align_celeba \
  --out-dir data/flow_cache \
  --method farneback

# Forensics flow cache (val/test) — run before Branch C eval
python data/precompute_flow.py \
  --img-dir data/forensics/images \
  --out-dir data/forensics_flow_cache \
  --method farneback
```

Cached as `{image_stem}_flow.pt` → shape `(2, 64, 64)` (dx, dy channels).

### Download Scripts

```bash
# Training data
kaggle datasets download -d jessicali9530/celeba-dataset
unzip celeba-dataset.zip -d data/celeba

# Val/test data
kaggle datasets download -d shivamardeshna/real-and-fake-images-dataset-for-image-forensics
unzip real-and-fake-images-dataset-for-image-forensics.zip -d data/forensics
```

---

## 7. Training Strategy (4 Phases)

### Phased Approach Rationale

Training all branches simultaneously from scratch leads to unstable gradients and branch co-adaptation. The phased freeze strategy ensures each branch learns strong independent features before the fusion head is trained.

### Hyperparameters

| Parameter              | Value                                                |
| ---------------------- | ---------------------------------------------------- |
| Image size             | 64×64                                                |
| Batch size             | 64                                                   |
| Optimizer              | Adam (β₁=0.5, β₂=0.999)                              |
| LR (phases 1–3)        | 2e-4                                                 |
| LR (phase 4 fine-tune) | 5e-5                                                 |
| Epochs                 | 20 for phases 1–3; 30 for Phase 4 staged fine-tuning |
| Scheduler              | CosineAnnealingLR                                    |
| Dropout (fusion head)  | 0.3                                                  |
| Fake ratio             | 0.5 (balanced)                                       |

### Phase Summary

| Phase | Trainable Parameters               | Frozen           | Target Metric        |
| ----- | ---------------------------------- | ---------------- | -------------------- |
| 1     | Branch A baseline + FC             | —                | Acc ≥ 77%, F1 ≥ 0.70 |
| 2     | Branch B + A+B fusion head         | Branch A encoder | Acc ≥ 88%, F1 ≥ 0.88 |
| 3     | Branch C + A+B+C fusion head       | Branch A, B      | Acc ≥ 83%, F1 ≥ 0.80 |
| 4     | Full fused model or ensemble stack | —                | Acc ≥ 94%, F1 ≥ 0.93 |

> All phase targets are evaluated against the **forensics val set**, not the CelebA proxy split.

---

## 8. Loss Functions

### Primary — Binary Cross-Entropy

```
L_BCE = −[ y · log D(x) + (1 − y) · log(1 − D(G(z))) ]
```

Simple, interpretable. Used for all phased training.

### Stability — Hinge Loss

```
L_hinge = E[max(0, 1 − D(x))] + E[max(0, 1 + D(G(z)))]
```

Enforces a margin between real and fake predictions under the proposal's discriminator notation.

### Combined (Phase 4)

```
L_total = α · L_BCE_weighted + (1 − α) · L_hinge_asymmetric      α = 0.7
```

The implemented Phase 4 loss uses the repository's fake-positive logit convention (`label 1 = fake`) and upweights real-class mistakes with `real_weight=1.5`. Its hinge term pushes real logits below `-margin` and fake logits above `margin`; the current default margin is `0.8`.

Final Phase 4 evaluation shows this loss did not fix the real/fake operating-point imbalance on the proxy task. TNR dropped from `0.78` to `0.72` while TPR rose from `0.94` to `0.97`, and F1 fell from `0.9072` to `0.8955`. Treat this as a characterized experiment, not the preferred deployment checkpoint.

---

## 9. Evaluation & Metrics

### Metrics

| Metric            | Description                                      |
| ----------------- | ------------------------------------------------ |
| Balanced Accuracy | Average of TPR and TNR — handles class imbalance |
| F1 Score          | Harmonic mean of precision and recall            |
| AUC-ROC           | Area under ROC curve                             |
| Confusion Matrix  | Per-class breakdown: authentic vs. synthetic     |

### Evaluation Dataset

Val and test evaluation uses the **Real & Fake Images Dataset for Image Forensics** (`shivamardeshna/real-and-fake-images-dataset-for-image-forensics`). This contains genuine AI-generated and manipulated fake images — not noise duplicates — making it a valid deepfake detection benchmark.

The forensics dataset is split 50/50 val/test, with real and fake images mixed at balanced ratio as specified by the team lead. The test set is held out until Week 4 final evaluation.

### Ensemble Strategy

For the B+C ensemble (recommended per proposal):

1. Train Branch B and Branch C independently to convergence
2. Extract their output logits or summary outputs as features
3. Fit a **Random Forest classifier** on [B_logit, C_logit] → real/fake
4. Optionally stack with Branch A logit for the full A+B+C ensemble

The current repository implements this as feature probes over the active Phase 3/4 branch outputs:

- `evaluation/ensemble.py` extracts A `2048-D`, B `32-D`, C `28-D`, and full-model logit features.
- Single-branch A/B/C configs use logistic regression.
- A+B, A+C, B+C, and A+B+C configs use `RandomForestClassifier(n_estimators=100, random_state=42)`.
- `scripts/run_ensemble_ablation.py` balances extracted examples by class, uses an 80/20 probe split, writes normalized and raw confusion matrices for all seven configs, and runs the Phase 3 threshold sweep in the same output directory.
- The completed proxy-task run in `runs/ensemble_ablation/` did not clear the B+C proposal gate, and the completed forensics OOD run in `runs/forensics_eval/` also fails the gate. No deployment claim is supported by the current artifacts.

---

## 10. Expected Performance Targets

| Configuration                    | Authentic % | Synthetic % | F1       | Notes                |
| -------------------------------- | ----------- | ----------- | -------- | -------------------- |
| Branch A only (CNN baseline)     | 77.8%       | 77.8%       | 0.70     | Phase 1 gate         |
| Branch B only (spatiotemporal)   | 88.9%       | 94.4%       | 0.91     |                      |
| Branch C only (physics dynamics) | 83.3%       | 83.3%       | 0.80     |                      |
| A + B ensemble                   | 89.5%       | 89.5%       | 0.88     |                      |
| A + C ensemble                   | 88.9%       | 88.9%       | 0.85     |                      |
| **B + C ensemble**               | **94.4%**   | **94.4%**   | **0.93** | ⭐ Proposal target   |
| A + B + C full ensemble          | 89.5%       | 89.5%       | 0.86     | Note: lower than B+C |

> **Current repo result:** B+C RF reached `0.8869` balanced accuracy on the proxy split and `0.4716` on the forensics OOD split — both below the proposal gate. A+B+C RF was the strongest proxy-task probe at `0.8992` balanced accuracy. The final report treats this as a negative transfer result. All targets above are the proposal's stated goals; the forensics test-set evaluation in Week 4 is the canonical measurement against these targets.

---

## 11. Risks & Mitigations

| Risk                                                          | Likelihood | Impact | Mitigation                                                              |
| ------------------------------------------------------------- | ---------- | ------ | ----------------------------------------------------------------------- |
| Branch A dominates training, suppresses B/C signal            | High       | High   | Phased freeze strategy; gradient scaling                                |
| Optical flow pre-computation bottleneck (~2h for full CelebA) | Medium     | Medium | One-time offline cache; skip during prototyping                         |
| Identity file missing (no pair sampling)                      | Low        | Medium | Fallback to adjacent-index pairs                                        |
| Real/fake class imbalance during GAN training                 | Medium     | Medium | Balanced sampler; monitor per-class accuracy                            |
| Overfitting on CelebA distribution                            | Medium     | High   | Forensics val/test set is the canonical OOD benchmark                   |
| Forensics flow cache pairing mismatch with identity loader    | Medium     | High   | Branch C must use consistent adjacent-index pairing or regenerate cache |
| CelebA license: non-commercial only                           | —          | —      | Confirm project usage is research-only                                  |

---

## 12. Dependencies

```
torch>=2.1.0
torchvision>=0.16.0
opencv-python>=4.8.0        # Farnebäck optical flow
numpy>=1.24.0
scikit-learn>=1.3.0         # Random forest ensemble
Pillow>=10.0.0
tqdm>=4.66.0
tensorboard>=2.14.0         # or wandb
pyyaml>=6.0
```

Install:

```bash
pip install -r requirements.txt
```

---

## References

1. Barrington, S. & Farid, H. (2026). _Distinguishing Authentic from AI-Generated Explosions using Spatiotemporal Dynamics._ CVPR Workshop 2026.
2. Internò, C. et al. (2025). _AI-Generated Video Detection via Perceptual Straightening._ arXiv:2507.00583.
3. Farnebäck, G. (2003). _Two-Frame Motion Estimation Based on Polynomial Expansion._ Image Analysis, Springer.
4. Miyato, T. et al. (2018). _Spectral Normalization for Generative Adversarial Networks._ ICLR 2018.
5. Goodfellow, I. et al. (2014). _Generative Adversarial Nets._ NeurIPS 2014.
6. Liu, Z. et al. (2015). _Deep Learning Face Attributes in the Wild._ ICCV 2015.
