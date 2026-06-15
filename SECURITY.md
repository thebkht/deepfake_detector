# Security Policy

## Supported versions

This is a research project under active development. Security fixes are applied to the `main` branch only; there are no separately maintained release branches.

| Version       | Supported |
| ------------- | --------- |
| `main`        | ✅        |
| older commits | ❌        |

## Reporting a vulnerability

Please report security vulnerabilities **privately**. Do not open a public GitHub issue for a security problem.

- Email: **me@thebkht.com**
- Alternatively, use GitHub's [private vulnerability reporting](https://github.com/thebkht/deepfake_detector/security/advisories/new) if enabled.

Please include:

- A description of the issue and its potential impact
- Steps to reproduce, or a proof of concept
- Any relevant environment details (OS, Python, PyTorch version)

## Response expectations

- Acknowledgement of your report within **7 days**.
- A status update on triage and remediation within **30 days**.
- Coordinated disclosure: please give us a reasonable window to release a fix before any public disclosure.

## Scope

This repository contains model training and evaluation code. Note that machine-learning models carry inherent risks (e.g. adversarial inputs, distribution shift, data-dependent behavior) that are research concerns rather than software vulnerabilities. Reports of genuine code-level security issues — for example unsafe deserialization, command injection in scripts, or path-traversal in data loaders — are in scope and very welcome.
