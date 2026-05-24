"""CelebA frame-pair dataset and dataloader factory for Week 1 and Phase 3."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from pickle import UnpicklingError
from typing import Callable, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, TypedDict, cast

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from data.augmentations import build_transforms


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_SPLIT_FRACTIONS = {
    "train": 0.8034284473269854,
    "val": 0.09806168835976585,
    "test": 0.09850986431324883,
}


def _resolve_image_dir(path_like: Path) -> Path:
    if (path_like / "img_align_celeba").is_dir():
        return path_like / "img_align_celeba"
    return path_like


def discover_celeba_images(image_dir: str | Path) -> List[Path]:
    root = _resolve_image_dir(Path(image_dir))
    if not root.exists():
        raise FileNotFoundError(f"CelebA image directory does not exist: {root}")
    images = sorted(
        path for path in root.iterdir() if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No image files found under: {root}")
    return images


def load_config(config_path: str | Path) -> Mapping[str, object]:
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    paths = dict(config.get("paths", {}))
    project_root_value = paths.get("project_root", ".")
    project_root = (config_path.parent / project_root_value).resolve()
    paths["project_root"] = str(project_root)

    for key, value in list(paths.items()):
        if key == "project_root":
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            paths[key] = str((project_root / candidate).resolve())

    config["paths"] = paths
    return config


def validate_celeba_dataset(
    image_dir: str | Path,
    expected_count: Optional[int] = None,
    expected_resolution: Optional[Tuple[int, int]] = None,
    sample_count: int = 3,
) -> Dict[str, object]:
    images = discover_celeba_images(image_dir)
    if expected_count is not None and len(images) != expected_count:
        raise ValueError(f"Expected {expected_count} images, found {len(images)}")

    sampled = images[: max(1, min(sample_count, len(images)))]
    resolutions = []
    for sample in sampled:
        with Image.open(sample) as img:
            resolutions.append(img.size)
            if expected_resolution is not None and img.size != expected_resolution:
                raise ValueError(
                    f"Expected resolution {expected_resolution} but found {img.size} in {sample.name}"
                )

    return {
        "image_dir": str(_resolve_image_dir(Path(image_dir))),
        "count": len(images),
        "sampled_resolutions": resolutions,
    }


@dataclass(frozen=True)
class PairMetadata:
    anchor_path: str
    pair_path: str
    pair_type: str
    pair_strategy: str
    identity: Optional[int]


class FramePairSample(TypedDict):
    frame_a: torch.Tensor
    frame_b: torch.Tensor
    label: torch.Tensor
    metadata: Dict[str, object]
    flow: NotRequired[torch.Tensor]


class PathsConfig(TypedDict):
    image_dir: str
    identity_file: NotRequired[str]


class DatasetConfig(TypedDict, total=False):
    image_size: int | float
    fake_ratio: int | float
    gaussian_noise_std: int | float
    train_split: int | float
    val_split: int | float
    test_split: int | float


class DataloaderConfig(TypedDict, total=False):
    batch_size: int
    num_workers: int
    pin_memory: bool
    drop_last: bool


class LoaderConfig(TypedDict):
    paths: PathsConfig
    dataset: DatasetConfig
    dataloader: DataloaderConfig
    phase3: NotRequired[Dict[str, object]]


PairingMode = Literal["default", "adjacent_cache"]


def _select_adjacent_partner_index(index: int, total_size: int) -> int:
    partner_index = min(index + 1, total_size - 1)
    if partner_index == index:
        partner_index = max(0, index - 1)
    return partner_index


def _load_flow_tensor(path: str | Path) -> torch.Tensor:
    flow_path = Path(path)
    load_errors = (RuntimeError, UnpicklingError, AttributeError, TypeError, ValueError)
    try:
        flow_tensor = torch.load(flow_path, map_location="cpu", weights_only=True)
    except load_errors:
        # Flow cache files are trusted local tensor dumps created by precompute_flow.py.
        flow_tensor = torch.load(flow_path, map_location="cpu", weights_only=False)
    if not isinstance(flow_tensor, torch.Tensor):
        raise TypeError(f"Expected flow cache tensor at {flow_path}, got {type(flow_tensor).__name__}")
    if tuple(flow_tensor.shape) != (2, 64, 64):
        raise ValueError(f"Expected flow tensor shape (2, 64, 64), got {tuple(flow_tensor.shape)}")
    if flow_tensor.dtype != torch.float32:
        flow_tensor = flow_tensor.float()
    if torch.isnan(flow_tensor).any():
        raise ValueError(f"Flow tensor contains NaN values: {flow_path}")
    return flow_tensor


def verify_flow_cache(image_dir: str | Path, cache_dir: str | Path) -> Dict[str, object]:
    images = discover_celeba_images(image_dir)
    cache_path = Path(cache_dir)
    expected_stems = {image_path.stem for image_path in images}
    actual_files = list(cache_path.glob("*_flow.pt")) if cache_path.exists() else []
    actual_stems = {path.name[: -len("_flow.pt")] for path in actual_files}
    missing_stems = sorted(expected_stems - actual_stems)
    extra_stems = sorted(actual_stems - expected_stems)
    return {
        "count": len(actual_files),
        "expected_count": len(expected_stems),
        "missing": missing_stems,
        "missing_count": len(missing_stems),
        "extra": extra_stems,
        "extra_count": len(extra_stems),
        "cache_dir": str(cache_path),
    }


class CelebAFramePairDataset(Dataset):
    """Stable Week 1 contract for real/fake frame-pair sampling.

    Real pairs:
    - With `identity_CelebA.txt` present, sample two images for the same identity.
    - Without the identity file, fall back to adjacent-index sampling.

    Fake pairs:
    - Pair an anchor image with a frame from a different identity when identity labels exist.
    - Without the identity file, fall back to a deterministic distant-index pairing.
    """

    def __init__(
        self,
        image_dir: str | Path,
        identity_file: str | Path | None = None,
        image_size: int = 64,
        fake_ratio: float = 0.5,
        gaussian_noise_std: float = 0.05,
        transform: Optional[Callable] = None,
        train: bool = True,
        limit: Optional[int] = None,
        flow_cache_dir: str | Path | None = None,
        pairing_mode: PairingMode = "default",
    ) -> None:
        self.image_paths = discover_celeba_images(image_dir)
        if limit is not None:
            self.image_paths = self.image_paths[:limit]
        self.index_by_name = {path.name: idx for idx, path in enumerate(self.image_paths)}
        self.image_size = image_size
        self.fake_ratio = fake_ratio
        self.gaussian_noise_std = gaussian_noise_std  # retained for API compatibility; unused after cross-identity fake strategy
        self.transform = transform or build_transforms(image_size=image_size, train=train)
        self.identity_file = Path(identity_file) if identity_file else None
        self.identity_lookup: Dict[str, int] = {}
        self.identity_groups: Dict[int, List[int]] = {}
        self.cross_identity_candidates: Dict[int, List[int]] = {}
        self.has_identity_file = False
        self._fake_fraction = Fraction(str(fake_ratio)).limit_denominator(1000)
        self.flow_cache_dir = Path(flow_cache_dir) if flow_cache_dir is not None else None
        self.pairing_mode = pairing_mode
        self._load_identity_pairs()

    def _load_identity_pairs(self) -> None:
        if not self.identity_file or not self.identity_file.exists():
            return

        groups: Dict[int, List[int]] = defaultdict(list)
        valid_names = {path.name for path in self.image_paths}
        with open(self.identity_file, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                filename, identity_value = parts
                if filename not in valid_names:
                    continue
                identity = int(identity_value)
                self.identity_lookup[filename] = identity
                groups[identity].append(self._index_for_filename(filename))

        self.identity_groups = {identity: sorted(indices) for identity, indices in groups.items()}
        self.has_identity_file = bool(self.identity_lookup)
        if self.has_identity_file:
            all_indices = list(range(len(self.image_paths)))
            self.cross_identity_candidates = {
                identity: [idx for idx in all_indices if self.identity_lookup.get(self.image_paths[idx].name) != identity]
                for identity in self.identity_groups
            }

    def _index_for_filename(self, filename: str) -> int:
        return self.index_by_name[filename]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, object]:
        if index < 0 or index >= len(self):
            raise IndexError(index)

        anchor_path = self.image_paths[index]
        flow_tensor: Optional[torch.Tensor] = None
        if self.flow_cache_dir is not None:
            flow_tensor = _load_flow_tensor(self.flow_cache_dir / f"{anchor_path.stem}_flow.pt")

        if self.pairing_mode == "adjacent_cache":
            sample = self._getitem_adjacent_cache(index=index, anchor_path=anchor_path)
            if flow_tensor is not None:
                sample["flow"] = flow_tensor
            return sample

        is_fake = self._is_fake_index(index)

        if is_fake:
            frame_a = self._load_tensor(anchor_path)
            fake_index, strategy = self._select_cross_identity_index(index)
            fake_path = self.image_paths[fake_index]
            frame_b = self._load_tensor(fake_path)
            metadata = PairMetadata(
                anchor_path=str(anchor_path),
                pair_path=str(fake_path),
                pair_type="fake",
                pair_strategy=strategy,
                identity=self.identity_lookup.get(anchor_path.name),
            )
            label = 1
        else:
            pair_index, identity_value, strategy = self._select_real_pair(index)
            pair_path = self.image_paths[pair_index]
            frame_a = self._load_tensor(anchor_path)
            frame_b = self._load_tensor(pair_path)
            metadata = PairMetadata(
                anchor_path=str(anchor_path),
                pair_path=str(pair_path),
                pair_type="real",
                pair_strategy=strategy,
                identity=identity_value,
            )
            label = 0

        sample: Dict[str, object] = {
            "frame_a": frame_a,
            "frame_b": frame_b,
            "label": torch.tensor(label, dtype=torch.long),
            "metadata": metadata.__dict__,
        }
        if flow_tensor is not None:
            sample["flow"] = flow_tensor
        return sample

    def _getitem_adjacent_cache(self, *, index: int, anchor_path: Path) -> Dict[str, object]:
        pair_index = self._select_adjacent_partner(index)
        pair_path = self.image_paths[pair_index]
        frame_a = self._load_tensor(anchor_path)
        frame_b = self._load_tensor(pair_path)
        anchor_identity = self.identity_lookup.get(anchor_path.name)
        pair_identity = self.identity_lookup.get(pair_path.name)
        if self.has_identity_file and anchor_identity is not None and pair_identity is not None:
            label = 0 if anchor_identity == pair_identity else 1
            pair_type = "real" if label == 0 else "fake"
            identity_value: Optional[int] = anchor_identity if label == 0 else None
            strategy = "adjacent_cache_identity_match"
        else:
            label = int(self._is_fake_index(index))
            pair_type = "real" if label == 0 else "fake"
            identity_value = anchor_identity
            strategy = "adjacent_cache_fallback_label"
        metadata = PairMetadata(
            anchor_path=str(anchor_path),
            pair_path=str(pair_path),
            pair_type=pair_type,
            pair_strategy=strategy,
            identity=identity_value,
        )
        return {
            "frame_a": frame_a,
            "frame_b": frame_b,
            "label": torch.tensor(label, dtype=torch.long),
            "metadata": metadata.__dict__,
        }

    def _is_fake_index(self, index: int) -> bool:
        if self._fake_fraction.numerator == 0:
            return False
        if self._fake_fraction.numerator >= self._fake_fraction.denominator:
            return True
        return (index % self._fake_fraction.denominator) < self._fake_fraction.numerator

    def _load_tensor(self, image_path: Path) -> torch.Tensor:
        with Image.open(image_path) as img:
            return self.transform(img.convert("RGB"))

    def _select_cross_identity_index(self, index: int) -> Tuple[int, str]:
        anchor_name = self.image_paths[index].name
        anchor_identity = self.identity_lookup.get(anchor_name)

        if self.has_identity_file and anchor_identity is not None:
            candidates = self.cross_identity_candidates.get(anchor_identity, [])
            if candidates:
                pick = candidates[index % len(candidates)]
                return pick, "cross_identity"

        offset = max(1, len(self.image_paths) // 2)
        pick = (index + offset) % len(self.image_paths)
        if pick == index and len(self.image_paths) > 1:
            pick = (index + 1) % len(self.image_paths)
        return pick, "distant_index_fallback"

    def _select_real_pair(self, index: int) -> Tuple[int, Optional[int], str]:
        anchor_name = self.image_paths[index].name
        identity_value = self.identity_lookup.get(anchor_name)
        if self.has_identity_file and identity_value is not None:
            group = self.identity_groups.get(identity_value, [])
            if len(group) >= 2:
                offset = group.index(index)
                pair_index = group[(offset + 1) % len(group)]
                return pair_index, identity_value, "same_identity"
            pair_index = self._select_adjacent_partner(index)
            return pair_index, identity_value, "identity_singleton_adjacent"

        pair_index = self._select_adjacent_partner(index)
        return pair_index, None, "adjacent_fallback"

    def _select_adjacent_partner(self, index: int) -> int:
        return _select_adjacent_partner_index(index, len(self.image_paths))


def collate_frame_pair_batch(batch: Sequence[FramePairSample]) -> Dict[str, object]:
    collated: Dict[str, object] = {
        "frame_a": torch.stack([sample["frame_a"] for sample in batch]),
        "frame_b": torch.stack([sample["frame_b"] for sample in batch]),
        "label": torch.stack([sample["label"] for sample in batch]),
        "metadata": [sample["metadata"] for sample in batch],
    }
    if all("flow" in sample for sample in batch):
        flow_tensors = [cast(torch.Tensor, sample.get("flow")) for sample in batch]
        collated["flow"] = torch.stack(flow_tensors)
    return collated


def _get_float(mapping: Mapping[str, object], key: str, default: float) -> float:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Expected numeric config value for '{key}', got {type(value).__name__}")
    return float(value)


def _get_int(mapping: Mapping[str, object], key: str) -> int:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Expected int config value for '{key}', got {type(value).__name__}")
    return value


def _get_bool(mapping: Mapping[str, object], key: str) -> bool:
    value = mapping[key]
    if not isinstance(value, bool):
        raise TypeError(f"Expected bool config value for '{key}', got {type(value).__name__}")
    return value


def _resolve_split_indices(total_size: int, dataset_cfg: Mapping[str, object], split: str) -> range:
    split_names = ("train", "val", "test")
    fractions = {
        "train": _get_float(dataset_cfg, "train_split", DEFAULT_SPLIT_FRACTIONS["train"]),
        "val": _get_float(dataset_cfg, "val_split", DEFAULT_SPLIT_FRACTIONS["val"]),
        "test": _get_float(dataset_cfg, "test_split", DEFAULT_SPLIT_FRACTIONS["test"]),
    }
    total_fraction = sum(fractions.values())
    if abs(total_fraction - 1.0) > 1e-6:
        raise ValueError("Dataset split fractions must sum to 1.0")
    if total_size <= 0:
        raise ValueError("Dataset must contain at least one sample")

    counts = {name: int(total_size * fractions[name]) for name in split_names}
    remainder = total_size - sum(counts.values())
    ranked_names = sorted(split_names, key=lambda name: fractions[name], reverse=True)
    for idx in range(remainder):
        counts[ranked_names[idx % len(ranked_names)]] += 1

    nonzero_splits = [name for name in split_names if fractions[name] > 0]
    if total_size >= len(nonzero_splits):
        for name in nonzero_splits:
            if counts[name] == 0:
                donor = max(
                    (candidate for candidate in nonzero_splits if counts[candidate] > 1),
                    key=lambda candidate: counts[candidate],
                )
                counts[donor] -= 1
                counts[name] += 1

    start = 0
    bounds: Dict[str, range] = {}
    for name in split_names:
        stop = start + counts[name]
        bounds[name] = range(start, stop)
        start = stop
    return bounds[split]


def create_celeba_dataloader(
    config: Mapping[str, object] | str | Path,
    split: str = "train",
    shuffle: Optional[bool] = None,
    limit: Optional[int] = None,
    include_flow: Optional[bool] = None,
    pairing_mode: Optional[PairingMode] = None,
) -> DataLoader:
    if isinstance(config, (str, Path)):
        config = load_config(config)
    typed_config = cast(LoaderConfig, config)

    paths = typed_config["paths"]
    dataset_cfg = typed_config["dataset"]
    dataloader_cfg = typed_config["dataloader"]
    phase3_cfg = cast(Mapping[str, object], typed_config.get("phase3", {}))
    split = split.lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    resolved_include_flow = bool(phase3_cfg.get("include_flow", False)) if include_flow is None else include_flow
    resolved_pairing_mode = cast(PairingMode, phase3_cfg.get("pairing_mode", "default")) if pairing_mode is None else pairing_mode
    dataset = CelebAFramePairDataset(
        image_dir=paths["image_dir"],
        identity_file=paths.get("identity_file"),
        image_size=_get_int(dataset_cfg, "image_size"),
        fake_ratio=_get_float(dataset_cfg, "fake_ratio", 0.5),
        gaussian_noise_std=_get_float(dataset_cfg, "gaussian_noise_std", 0.05),
        train=(split == "train"),
        limit=limit,
        flow_cache_dir=paths.get("flow_cache_dir") if resolved_include_flow else None,
        pairing_mode=resolved_pairing_mode,
    )
    split_indices = list(_resolve_split_indices(len(dataset), dataset_cfg, split))
    dataset = Subset(dataset, split_indices)

    if shuffle is None:
        shuffle = split == "train"

    return DataLoader(
        dataset,
        batch_size=_get_int(dataloader_cfg, "batch_size"),
        shuffle=shuffle,
        num_workers=_get_int(dataloader_cfg, "num_workers"),
        pin_memory=_get_bool(dataloader_cfg, "pin_memory"),
        drop_last=_get_bool(dataloader_cfg, "drop_last"),
        collate_fn=collate_frame_pair_batch,
    )
