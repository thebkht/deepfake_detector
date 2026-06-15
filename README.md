# Hybrid Three-Branch Deepfake Detector

[![CI](https://github.com/thebkht/deepfake_detector/actions/workflows/ci.yml/badge.svg)](https://github.com/thebkht/deepfake_detector/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)

A research implementation of a hybrid discriminator for deepfake face detection that fuses three complementary signals over consecutive face frames:

- **Branch A — spatial:** a CNN encoder over single frames (`2048-D`)
- **Branch B — spatiotemporal:** velocity/derivative statistics across a frame pair (`8-D`, expanded to `32-D`)
- **Branch C — physics:** deterministic optical-flow and photometric dynamics (`28-D`)

The branches are concatenated (`2048 + 32 + 28 = 2108`) and trained in four sequential phases, each building on the previous one's frozen weights.

> [!IMPORTANT]
> **Scope & intended use.** This is a research and educational project for **defensive media forensics**. It is a *detector* and does not generate synthetic media. The current training signal is a **proxy task** ("same identity vs. different identity" image pairs from CelebA), not real generative manipulations, and out-of-domain transfer to real forensics data currently **does not generalize**. Treat the numbers here as a reproducible baseline, not a production deepfake classifier. Do not use it to make consequential decisions about individuals, and respect the license/terms of any dataset you use.

## Architecture

The detector operates on consecutive `64 × 64` RGB face frames. Source lives in [`models/`](models/), with training orchestration in [`training/`](training/).

| Branch | Module | Output | Description |
| ------ | ------ | ------ | ----------- |
| A — spatial | [`models/branch_a.py`](models/branch_a.py) | `2048-D` | Five spectral-normalized conv blocks per frame. `BranchABaseline` classifies the concatenated `4096-D` pair. |
| B — spatiotemporal | [`models/branch_b.py`](models/branch_b.py) | `8 → 32-D` | Reuses the pretrained `BranchAEncoder` and derives an `8-D` velocity summary, normalized and expanded to a learned `32-D` feature. |
| C — physics | [`models/branch_c.py`](models/branch_c.py) | `28-D` | Deterministic extractor: `20-D` optical-flow summaries + `8-D` HSV photometric summaries. |

Fusion and the phased discriminators (`DiscriminatorPhase2/3/4`) live in [`models/discriminator.py`](models/discriminator.py). Training proceeds in phases:

1. **Phase 1** — train Branch A end-to-end.
2. **Phase 2** — add Branch B; Branch A is frozen except the last two blocks (fused at `2048 + 32`).
3. **Phase 3** — add Branch C with A+B frozen (fused at `2108`). Requires a precomputed optical-flow cache and `adjacent_cache` frame pairing.
4. **Phase 4** — fine-tune the fused model with staged unfreezing and an asymmetric loss.

> **Note on conventions:** the label convention is **fake-positive** (`fake = 1`), and Phase 3/4 must use `pairing_mode="adjacent_cache"` so cached flow tensors align with the correct frame pair. See [CONTRIBUTING.md](CONTRIBUTING.md) for the invariants to preserve when modifying the pipeline.

## Installation

Requires Python 3.11 or 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset

The pipeline trains on [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset). A bootstrap script downloads and extracts it:

```bash
bash scripts/download_celeba.sh
```

This requires the Kaggle CLI and credentials at `~/.kaggle/kaggle.json`. The loader ([`data/celeba_loader.py`](data/celeba_loader.py)) builds real pairs (same identity) and fake pairs (different identity) when `identity_CelebA.txt` is present, falling back to index-based pairing otherwise.

Branch C / Phase 3+ additionally require a precomputed optical-flow cache:

```bash
python3 -m data.precompute_flow \
  --img-dir data/celeba/img_align_celeba \
  --out-dir data/flow_cache \
  --method farneback \
  --image-size 64
```

## Usage

Each phase has a trainer module. Common flags: `--run-name`, `--device cpu|cuda|mps`, `--epochs-override`, and smoke-run caps (`--train-limit`, `--val-limit`, `--max-batches`). Defaults are configured per phase in [`config/config.yaml`](config/config.yaml).

```bash
# Phase 1 — Branch A baseline
python3 -m training.phase1_train --config config/config.yaml --run-name phase1

# Phase 2 — A + B
python3 -m training.phase2_train --config config/config.yaml --run-name phase2_a_b

# Phase 3 — A + B + C  (needs the flow cache above)
python3 -m training.phase3_train --config config/config.yaml --run-name phase3_a_b_c

# Phase 4 — fine-tune the fused model
python3 -m training.phase4_finetune --config config/config.yaml --run-name phase4
```

Each run writes checkpoints to `checkpoints/` and metrics, confusion matrices, preview grids, and curves to `runs/<run-name>/`.

### Evaluation

```bash
# Branch A test-split evaluation
python3 -m training.eval_branch_a --config config/config.yaml --run-name eval_branch_a

# RF ensemble + per-branch ablation + threshold sweep
python3 scripts/run_ensemble_ablation.py \
  --config config/config.yaml \
  --checkpoint checkpoints/phase3_a_b_c.pt \
  --run-dir runs/ensemble_ablation --device cpu

# Phase 3 decision-threshold sweep only
python3 -m evaluation.threshold_sweep \
  --config config/config.yaml \
  --checkpoint checkpoints/phase3_a_b_c.pt \
  --run-dir runs/threshold_sweep --device cpu
```

The ensemble evaluator extracts frozen A/B/C features and trains seven branch-combination probes (single-branch logistic + A+B / A+C / B+C / A+B+C random forests), reporting balanced accuracy, F1, and AUC-ROC with confusion matrices.

## Results

These are **in-domain proxy-task** numbers (same-vs-different identity on CelebA), reported for reproducibility — not deepfake-detection performance.

**Branch-combination probes** (balanced test subset):

| Config | Classifier | Bal. Acc | F1     | AUC-ROC |
| ------ | ---------- | -------- | ------ | ------- |
| A      | Logistic   | 0.6529   | 0.6224 | 0.6860  |
| B      | Logistic   | 0.8930   | 0.8897 | 0.9471  |
| C      | Logistic   | 0.5530   | 0.5516 | 0.5717  |
| A+B    | RF         | 0.8939   | 0.8913 | 0.9463  |
| A+C    | RF         | 0.6636   | 0.6338 | 0.6966  |
| B+C    | RF         | 0.8869   | 0.8837 | 0.9440  |
| A+B+C  | RF         | 0.8992   | 0.8962 | 0.9471  |

Among the neural checkpoints, **Phase 3 is the strongest deployment candidate** under a balanced objective (Phase 4's asymmetric fine-tuning improved fake recall but degraded real-class specificity and F1).

**Out-of-domain transfer** to real forensics images currently fails — every CelebA-trained ensemble collapses toward chance:

| Config | Bal. Acc | F1     | AUC-ROC |
| ------ | -------- | ------ | ------- |
| A      | 0.5014   | 0.6683 | 0.5344  |
| B      | 0.4683   | 0.4764 | 0.4559  |
| B+C    | 0.4716   | 0.4981 | 0.4572  |
| A+B+C  | 0.4843   | 0.5633 | 0.4635  |

The takeaway: CelebA identity-pair features do not transfer to real manipulated-face content. Closing this gap is the focus of the roadmap below.

## Project layout

```text
config/        config.yaml — per-phase hyperparameters and paths
data/          CelebA loader, augmentations, optical-flow precompute, face alignment
models/        branch_a/b/c.py and discriminator.py (phased fusion heads)
training/      phaseN_train.py CLIs + trainers, losses, checkpointing, early-stop, artifacts
evaluation/    metrics, RF ensemble, threshold sweep, forensics/OOD evaluation
scripts/       dataset download, ablation and forensics runners
tests/         unittest suite (self-contained; no dataset or checkpoint required)
```

## Testing

The suite is self-contained — it synthesizes fixtures in temp directories and needs no dataset or trained checkpoint.

```bash
python -m unittest discover -s tests          # full suite
python -m unittest tests.test_model           # a single module
```

CI runs the same suite on Python 3.11 and 3.12 for every push and pull request.

## Limitations

- The training signal is a **proxy task** (cross-identity pairs), not real generative deepfakes, so in-domain metrics do not reflect real detection performance.
- Out-of-domain transfer to forensics data currently fails the deployment gate for all branch combinations.
- Phase 4's asymmetric fine-tuning worsened the precision/recall balance; Phase 3 remains the recommended checkpoint.
- Phase 3/4 evaluation must keep `pairing_mode="adjacent_cache"` — other pairing modes attach cached flow tensors to the wrong frame pair unless the cache is regenerated.
- If `identity_CelebA.txt` is missing, pairing falls back to index-based heuristics; a locally supplied pseudo-identity file may be attribute-derived rather than true identity labels.

## Roadmap

1. Replace cross-identity proxy negatives with true manipulated-face training sources.
2. Add domain adaptation, or train transfer classifiers on forensics-like manipulations rather than CelebA identity-pair features alone.
3. Re-run the forensics OOD protocol after retraining.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, test commands, and the invariants to preserve, and the [Code of Conduct](CODE_OF_CONDUCT.md). Report security issues privately per [SECURITY.md](SECURITY.md).

## Citation

If you use this work, please cite it using the metadata in [CITATION.cff](CITATION.cff).

## License

Released under the MIT License — see [LICENSE](LICENSE).
