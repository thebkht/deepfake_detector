"""Batch preview image writers for training and validation splits."""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Optional

import torch
from PIL import Image, ImageDraw


_MAX_PREVIEW_BATCHES = 3
_TILE_SIZE = 128
_GRID_COLUMNS = 4
_PADDING = 4
_TEXT_HEIGHT = 18
_BACKGROUND_COLOR = (255, 255, 255)
_TEXT_COLOR = (0, 102, 204)


def _denormalize_image(image: torch.Tensor) -> torch.Tensor:
    image = image.detach().cpu().float()
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected image tensor in CHW RGB format")
    image = image.clamp(-1.0, 1.0)
    return ((image + 1.0) / 2.0).clamp(0.0, 1.0)


def _to_pil_image(image: torch.Tensor) -> Image.Image:
    denormalized = _denormalize_image(image)
    array = (denormalized.mul(255.0).byte().permute(1, 2, 0).numpy())
    pil_image = Image.fromarray(array, mode="RGB")
    return pil_image.resize((_TILE_SIZE, _TILE_SIZE))


def _compose_tile(image: torch.Tensor, caption: str) -> Image.Image:
    tile = Image.new("RGB", (_TILE_SIZE, _TILE_SIZE + _TEXT_HEIGHT), color=_BACKGROUND_COLOR)
    tile.paste(_to_pil_image(image), (0, _TEXT_HEIGHT))
    draw = ImageDraw.Draw(tile)
    draw.text((4, 2), caption, fill=_TEXT_COLOR)
    return tile


def _build_grid(images: list[torch.Tensor], captions: list[str]) -> Image.Image:
    if not images:
        raise ValueError("Expected at least one image to build a preview grid")
    if len(images) != len(captions):
        raise ValueError("Image and caption counts must match")

    rows = ceil(len(images) / _GRID_COLUMNS)
    grid_width = (_GRID_COLUMNS * _TILE_SIZE) + ((_GRID_COLUMNS - 1) * _PADDING)
    grid_height = (rows * (_TILE_SIZE + _TEXT_HEIGHT)) + ((rows - 1) * _PADDING)
    grid = Image.new("RGB", (grid_width, grid_height), color=_BACKGROUND_COLOR)

    for index, (image, caption) in enumerate(zip(images, captions)):
        row = index // _GRID_COLUMNS
        column = index % _GRID_COLUMNS
        x_offset = column * (_TILE_SIZE + _PADDING)
        y_offset = row * (_TILE_SIZE + _TEXT_HEIGHT + _PADDING)
        grid.paste(_compose_tile(image, caption), (x_offset, y_offset))

    return grid


def _label_name(label_value: int) -> str:
    return "fake" if label_value == 1 else "real"


def _prediction_caption(logit: torch.Tensor, label_value: int) -> str:
    probability = float(torch.sigmoid(logit.detach().cpu()).item())
    predicted_label = 1 if probability >= 0.5 else 0
    return f"{_label_name(predicted_label)} {probability:.1f}"


def _label_caption(label_value: int) -> str:
    return _label_name(label_value)


def _train_caption(label_value: int) -> str:
    return str(label_value)


def _save_grid(path: Path, images: list[torch.Tensor], captions: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _build_grid(images, captions).save(path, format="JPEG", quality=95)


def maybe_save_train_preview(
    *,
    run_dir: str | Path,
    batch_index: int,
    frame_a: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    if batch_index >= _MAX_PREVIEW_BATCHES:
        return

    output_path = Path(run_dir) / f"train_batch{batch_index}.jpg"
    if output_path.exists():
        return

    image_count = min(len(frame_a), 16)
    images = [frame_a[item_index] for item_index in range(image_count)]
    captions = [_train_caption(int(labels[item_index].item())) for item_index in range(image_count)]
    _save_grid(output_path, images, captions)


def maybe_save_val_previews(
    *,
    run_dir: str | Path,
    batch_index: int,
    frame_a: torch.Tensor,
    labels: torch.Tensor,
    logits: torch.Tensor,
) -> None:
    if batch_index >= _MAX_PREVIEW_BATCHES:
        return

    labels_path = Path(run_dir) / f"val_batch{batch_index}_labels.jpg"
    pred_path = Path(run_dir) / f"val_batch{batch_index}_pred.jpg"
    if labels_path.exists() and pred_path.exists():
        return

    image_count = min(len(frame_a), 16)
    images = [frame_a[item_index] for item_index in range(image_count)]

    if not labels_path.exists():
        label_captions = [_label_caption(int(labels[item_index].item())) for item_index in range(image_count)]
        _save_grid(labels_path, images, label_captions)

    if not pred_path.exists():
        pred_captions = [
            _prediction_caption(logits[item_index], int(labels[item_index].item())) for item_index in range(image_count)
        ]
        _save_grid(pred_path, images, pred_captions)


__all__ = ["maybe_save_train_preview", "maybe_save_val_previews"]
