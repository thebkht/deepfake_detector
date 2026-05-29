# Week 4 Ensemble Results

## Status

Week 4 Dev 1 evaluation support is implemented for the local forensics layout under `data/forensics/`. The current forensics numbers below are a smoke validation run, not the final full-run table: they use `--limit 128` per dataset and the smoke CelebA transfer cache at `runs/celeba_features/phase3_train_adjacent_cache_smoke.npz`.

The full deployment table remains blocked on the full CelebA train feature cache at `runs/celeba_features/phase3_train_adjacent_cache.npz`, followed by an unrestricted forensics run.

## In-Domain Proxy

Source: `runs/ensemble_ablation/summary.md`, balanced CelebA proxy test subset, `13,074` examples. The RF probe trained on an in-memory 80/20 split of test-split branch features, so it is an in-domain proxy ablation rather than the forensics transfer protocol.

| # | Config | Classifier | Balanced accuracy | F1 | AUC-ROC |
| -: | ------ | ---------- | ----------------: | --: | ------: |
| 1 | A only | Logistic | 0.6529 | 0.6224 | 0.6860 |
| 2 | B only | Logistic | 0.8930 | 0.8897 | 0.9471 |
| 3 | C only | Logistic | 0.5530 | 0.5516 | 0.5717 |
| 4 | A+B | RF | 0.8939 | 0.8913 | 0.9463 |
| 5 | A+C | RF | 0.6636 | 0.6338 | 0.6966 |
| 6 | B+C | RF | 0.8869 | 0.8837 | 0.9440 |
| 7 | A+B+C | RF | 0.8992 | 0.8962 | 0.9471 |

The B+C proposal gate is not cleared on this proxy table: balanced accuracy is `0.8869` versus the `0.944` target, and F1 is `0.8837` versus the `0.93` target.

## Phase 3 Neural Threshold

Source: `runs/ensemble_ablation/threshold_sweep.json`.

| Threshold | Balanced accuracy | F1 | TPR | TNR |
| --------: | ----------------: | --: | --: | --: |
| 0.61 | 0.8850 | 0.8808 | 0.8501 | 0.9198 |

Forensics neural evaluation reports both default threshold `0.5` and the proxy-selected `0.61` threshold.

## Forensics Transfer Smoke

Source: `runs/forensics_eval_smoke_all/summary.md`.

Command:

```bash
.venv-mps/bin/python scripts/run_forensics_eval.py \
  --config config/config.yaml \
  --checkpoint checkpoints/phase3_a_b_c.pt \
  --forensics-root data/forensics \
  --device cpu \
  --limit 128 \
  --max-batches 4 \
  --celeba-features runs/celeba_features/phase3_train_adjacent_cache_smoke.npz \
  --run-dir runs/forensics_eval_smoke_all
```

This run validates the artifact schema, the `adjacent_same_class` pair contract, on-the-fly Farneback flow, per-image CSV output, and all seven ensemble combo paths. Each dataset contributed `64` real and `64` fake images.

| Dataset | N | Neural bal acc @0.5 | Neural bal acc @0.61 | B+C transfer bal acc | A+B+C transfer bal acc |
| ------- | -: | ------------------: | -------------------: | -------------------: | ---------------------: |
| Data Set 1 | 128 | 0.5547 | 0.5078 | 0.4531 | 0.4453 |
| Data Set 2 | 128 | 0.4922 | 0.4766 | 0.4531 | 0.4297 |
| Data Set 3 | 128 | 0.4219 | 0.3984 | 0.4141 | 0.4688 |
| Data Set 4 | 128 | 0.5234 | 0.5547 | 0.5000 | 0.5000 |
| Pooled | 512 | 0.4980 | 0.4844 | 0.4551 | 0.4609 |

These are smoke-run numbers only. They should not be used as the final forensics gate result because the transfer RF was trained on a 206-row CelebA smoke cache.

## Deployment Recommendation

Recommended deployment configuration remains B+C RF as the architectural candidate: Branch B carries the strongest proxy signal, Branch C preserves the flow/physics channel, and Branch A is known to add in-distribution face-identity bias that can dilute OOD behavior. This is a recommendation about deployment shape, not a claim that the metric gate is cleared.

Gate status:

- Proxy gate: not cleared, B+C balanced accuracy `0.8869` vs target `0.944`.
- Forensics gate: not yet measured on the full transfer protocol.
- Neural baseline: use Phase 3 as the balanced deployment baseline; Phase 4 is characterized as a comparison checkpoint, not the preferred default.

## Architecture Checklist

- Automated architecture contracts live in `tests/test_architecture_contracts.py`.
- Phase 3 and Phase 4 forward paths return `a`, `b`, `c`, and `logit` with dimensions `2048`, `32`, `28`, and `(B,)`.
- The inference contract at `runs/phase4_ensemble/inference_contract.json` sums to `2108`.
- Branch B and Branch C normalization guards are covered by finite-output tests.
- Phase freeze behavior remains covered by the existing `tests/test_model.py` Phase 2/3/4 tests.
- No training imports are required inside `models/`.
- `adjacent_cache` remains CelebA-only where the flow cache matches adjacent image partners.
- Forensics evaluation intentionally uses on-the-fly Farneback flow because the forensics folders do not provide a CelebA-style flow cache.

## Full-Run Checklist

1. Extract the full CelebA train cache:

```bash
.venv-mps/bin/python scripts/extract_celeba_features.py \
  --config config/config.yaml \
  --checkpoint checkpoints/phase3_a_b_c.pt \
  --split train \
  --pairing-mode adjacent_cache \
  --device mps \
  --out runs/celeba_features/phase3_train_adjacent_cache.npz
```

2. Run unrestricted forensics transfer evaluation:

```bash
.venv-mps/bin/python scripts/run_forensics_eval.py \
  --config config/config.yaml \
  --checkpoint checkpoints/phase3_a_b_c.pt \
  --phase4-checkpoint checkpoints/phase4_ensemble.pt \
  --forensics-root data/forensics \
  --split test \
  --device mps \
  --celeba-features runs/celeba_features/phase3_train_adjacent_cache.npz \
  --run-dir runs/forensics_eval
```

3. Replace the smoke table above with `runs/forensics_eval/summary.md` values and update `docs/build-plan.md` once the full numbers exist.
