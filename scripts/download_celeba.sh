#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data/celeba"
ZIP_PATH="${DATA_DIR}/img_align_celeba.zip"
EXTRACT_DIR="${DATA_DIR}/img_align_celeba"
DATASET_REF="jessicali9530/celeba-dataset"

mkdir -p "${DATA_DIR}"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "Error: kaggle CLI is not installed." >&2
  echo "Install it with: pip install kaggle" >&2
  exit 1
fi

if [[ ! -f "${HOME}/.kaggle/kaggle.json" ]]; then
  echo "Error: Kaggle credentials not found at ~/.kaggle/kaggle.json" >&2
  exit 1
fi

if [[ ! -f "${ZIP_PATH}" ]]; then
  echo "Downloading CelebA archive from Kaggle..."
  kaggle datasets download \
    --dataset "${DATASET_REF}" \
    --file img_align_celeba.zip \
    --path "${DATA_DIR}"
else
  echo "Archive already exists: ${ZIP_PATH}"
fi

if [[ -d "${EXTRACT_DIR}/img_align_celeba" ]]; then
  echo "Extracted dataset already exists: ${EXTRACT_DIR}/img_align_celeba"
  exit 0
fi

echo "Extracting archive..."
unzip -q -o "${ZIP_PATH}" -d "${EXTRACT_DIR}"

echo "CelebA is ready at: ${EXTRACT_DIR}/img_align_celeba"
