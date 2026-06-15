# Contributing

Thanks for your interest in improving the Hybrid Three-Branch Deepfake Detector. This is a research codebase, so contributions that improve correctness, reproducibility, documentation, and out-of-domain generalization are especially welcome.

## Ground rules

By participating you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

Before investing significant effort, please open an issue to discuss the change — particularly for anything that touches the model architecture, the fusion contract, or the training pipeline.

## Project context you should know first

A few invariants in this codebase are load-bearing. Breaking them silently invalidates checkpoints or metrics:

- **The active fusion contract is `2048 + 32 + 28 = 2108`**, not the proposal's `2048 + 8 + 28 = 2084`. The `32` is Branch B's learned expansion of its `8`-D summary. Existing checkpoints are only load-compatible with `2108`.
- **Phase 3 and Phase 4 must use `pairing_mode="adjacent_cache"`.** Cached optical-flow tensors are keyed to adjacent frame pairs; other pairing modes attach the wrong flow unless the cache is regenerated.
- **The label convention is fake-positive** (`fake = 1`). The hinge and asymmetric losses depend on this.
- **The current training task is a proxy** ("same identity vs. different identity"), not real generative deepfakes. Do not present in-domain proxy metrics as deepfake-detection results.

See `README.md` for the full architecture and status, and `CLAUDE.md` for an orientation map.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The test suite is self-contained — it synthesizes its own fixtures in temporary directories and does not require the CelebA dataset or any trained checkpoint.

```bash
python -m unittest discover -s tests          # full suite
python -m unittest tests.test_model           # single module
python -m unittest tests.test_data.TestClass.test_method   # single case
```

## Submitting changes

1. Fork the repository and create a feature branch off `main`.
2. Make your change with tests. New behavior should come with a test; bug fixes should come with a regression test.
3. Run the full suite locally and make sure it passes. CI runs the same suite on every pull request.
4. Keep your code consistent with the surrounding style — match existing naming, type hints, and module layout. The repo type-checks under Pyright in `basic` mode (`pyrightconfig.json`).
5. Open a pull request describing the motivation, the approach, and any metric or contract implications. Link the issue it addresses.

## Reporting bugs and requesting features

Use the GitHub issue templates. For bugs, include the command you ran, the device (`cpu`/`cuda`/`mps`), your Python and PyTorch versions, and the full traceback. For training/evaluation issues, include the relevant config block and run name.
