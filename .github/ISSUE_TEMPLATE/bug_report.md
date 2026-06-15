---
name: Bug report
about: Report a problem with training, evaluation, or the model code
title: "[Bug] "
labels: bug
assignees: ''
---

## Description

A clear and concise description of what the bug is.

## To reproduce

The exact command you ran, e.g.:

```bash
python3 -m training.phase3_train --config config/config.yaml --run-name repro --device cpu
```

## Expected behavior

What you expected to happen.

## Traceback / logs

```
<paste the full traceback or relevant log output here>
```

## Environment

- OS:
- Device: <cpu / cuda / mps>
- Python version: <output of `python --version`>
- PyTorch version: <output of `python -c "import torch; print(torch.__version__)"`>
- Branch / commit:

## Config and run context

- Relevant `config/config.yaml` block (e.g. `phase3:`), if applicable:
- Run name / checkpoint involved:

## Additional context

Anything else that might help — dataset state, flow-cache state, whether `identity_CelebA.txt` is present, etc.
