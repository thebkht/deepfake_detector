"""Face alignment helpers for forensics OOD preprocessing."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

from PIL import Image


def align_face(
    image: Image.Image,
    *,
    margin: float = 0.25,
    image_size: int = 64,
    device: str = "cpu",
    mtcnn: Optional[Any] = None,
) -> Image.Image | None:
    """Return an MTCNN-aligned face crop, or ``None`` when no face is found."""
    rgb = image.convert("RGB")
    detector = mtcnn if mtcnn is not None else _get_mtcnn(image_size=image_size, margin=margin, device=device)
    aligned = _run_detector(detector, rgb, device=device if mtcnn is None else "cpu", image_size=image_size, margin=margin)
    if aligned is None:
        return None
    if isinstance(aligned, Image.Image):
        return aligned.convert("RGB")
    try:
        import torch
        from torchvision.transforms.functional import to_pil_image
    except ModuleNotFoundError as error:
        raise RuntimeError("torch and torchvision are required for tensor MTCNN output") from error
    if isinstance(aligned, torch.Tensor):
        tensor = aligned.detach().cpu()
        if tensor.ndim == 4:
            tensor = tensor[0]
        # MTCNN with post_process=False returns float32 in [0, 255].
        # MTCNN with post_process=True  returns float32 in [-1, 1].
        # Detect the range and normalise to [0, 1] for to_pil_image.
        if float(tensor.max()) > 1.0:
            tensor = tensor / 255.0
        elif float(tensor.min()) < 0.0:
            tensor = (tensor + 1.0) / 2.0
        return to_pil_image(tensor.clamp(0.0, 1.0)).convert("RGB")
    raise TypeError(f"Unsupported MTCNN output type: {type(aligned)!r}")


def _run_detector(detector: Any, rgb: Image.Image, *, device: str, image_size: int, margin: float) -> Any:
    """Run MTCNN, falling back to a CPU detector on MPS adaptive-pool errors.

    Some image sizes cause ``AdaptiveAvgPool2d`` to fail on MPS because MPS
    requires input dimensions to be evenly divisible by output dimensions.
    When that happens we transparently retry with a CPU-bound detector so the
    image is not silently skipped.
    """
    try:
        return detector(rgb)
    except RuntimeError as exc:
        if device != "mps" or "Adaptive pool MPS" not in str(exc):
            raise
        cpu_detector = _get_mtcnn(image_size=image_size, margin=margin, device="cpu")
        return cpu_detector(rgb)


def align_face_or_fallback(
    image: Image.Image,
    *,
    margin: float = 0.25,
    image_size: int = 64,
    fallback: str = "center_crop",
    device: str = "cpu",
    mtcnn: Optional[Any] = None,
) -> Image.Image:
    """Align a face when possible, otherwise return a deterministic fallback crop."""
    aligned = align_face(image, margin=margin, image_size=image_size, device=device, mtcnn=mtcnn)
    if aligned is not None:
        return aligned
    if fallback != "center_crop":
        raise ValueError(f"Unsupported face alignment fallback: {fallback}")
    return _center_crop_square(image.convert("RGB")).resize((image_size, image_size), Image.Resampling.BILINEAR)


@lru_cache(maxsize=8)
def _get_mtcnn(*, image_size: int, margin: float, device: str) -> Any:
    try:
        from facenet_pytorch import MTCNN  # type: ignore[import-not-found]
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "facenet-pytorch is required for MTCNN alignment. "
            "Install requirements.txt or pass a mocked mtcnn in tests."
        ) from error
    pixel_margin = max(0, int(round(float(margin) * int(image_size))))
    return MTCNN(image_size=int(image_size), margin=pixel_margin, post_process=False, device=device)


def _center_crop_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


__all__ = ["align_face", "align_face_or_fallback"]