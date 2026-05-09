from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("pyyaml is required. Install with `pip install pyyaml`.") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
MODALITIES = ("param_only", "visual_only", "fusion")


@dataclass(frozen=True)
class RunPaths:
    clip_manifest: Path
    target_manifest: Path
    target_stats: Path
    run_config_snapshot: Path
    processed_root: Path
    targets_root: Path
    model_inputs_root: Path
    evaluation_root: Path


def fail(message: str) -> None:
    raise RuntimeError(message)


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        fail(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        fail(f"Invalid YAML top-level type in {path}")
    return data


def run_suffix(run_id: str | None) -> str:
    rid = (run_id or "").strip()
    if rid == "" or rid.lower() == "canonical":
        return ""
    return f"_{rid}"


def with_run_suffix(stem: str, suffix: str, ext: str) -> str:
    return f"{stem}{suffix}{ext}"


def split_filename(split: str, run_id: str | None, ext: str) -> str:
    suffix = run_suffix(run_id)
    return f"{split}{suffix}{ext}" if suffix else f"{split}{ext}"


def resolve_run_paths(config: Dict[str, Any], run_id: str | None) -> RunPaths:
    path_cfg = config.get("paths", {})
    if not isinstance(path_cfg, dict):
        fail("config.yaml missing `paths` mapping.")

    manifest_dir = PROJECT_ROOT / str(path_cfg.get("manifest_dir", "data/interim/manifests"))
    processed_root = PROJECT_ROOT / str(path_cfg.get("processed_dir", "data/processed"))
    suffix = run_suffix(run_id)

    return RunPaths(
        clip_manifest=manifest_dir / "clip_manifest.csv",
        target_manifest=manifest_dir / with_run_suffix("target_manifest", suffix, ".csv"),
        target_stats=manifest_dir / with_run_suffix("target_stats", suffix, ".csv"),
        run_config_snapshot=manifest_dir / with_run_suffix("run_config_snapshot", suffix, ".json"),
        processed_root=processed_root,
        targets_root=processed_root / "targets",
        model_inputs_root=processed_root / "model_inputs",
        evaluation_root=processed_root / "evaluation",
    )


def resolve_paths(config_path: str | Path | None = None, run_id: str | None = None) -> RunPaths:
    cfg_path = PROJECT_ROOT / "configs" / "config.yaml" if config_path is None else Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    return resolve_run_paths(load_yaml(cfg_path), run_id)


def get_fingerprint_path(paths: RunPaths, run_id: str | None) -> Path:
    suffix = run_suffix(run_id)
    return paths.model_inputs_root / with_run_suffix("cache_fingerprint", suffix, ".json")


def get_learning_schema_path(paths: RunPaths, run_id: str | None) -> Path:
    suffix = run_suffix(run_id)
    return paths.model_inputs_root / with_run_suffix("learning_schema", suffix, ".json")


def get_normalization_stats_path(paths: RunPaths, run_id: str | None) -> Path:
    suffix = run_suffix(run_id)
    return paths.model_inputs_root / "fusion" / with_run_suffix("normalization_stats", suffix, ".json")


def get_surrogate_targets_path(paths: RunPaths, run_id: str | None) -> Path:
    suffix = run_suffix(run_id)
    return paths.targets_root / with_run_suffix("surrogate_targets", suffix, ".csv")


def get_shared_sample_manifest_path(paths: RunPaths, run_id: str | None) -> Path:
    suffix = run_suffix(run_id)
    return paths.model_inputs_root / with_run_suffix("shared_sample_manifest", suffix, ".json")


def get_split_index_path(paths: RunPaths, run_id: str | None, modality: str, split: str) -> Path:
    return paths.model_inputs_root / modality / split_filename(split, run_id, ".pt")


def shared_cache_dir(paths: RunPaths, run_id: str | None, clip_len: int, img_size: int, cache_dtype: str) -> Path:
    suffix = run_suffix(run_id)
    tag = f"shared_video_cache{suffix}_T{clip_len}_S{img_size}_{cache_dtype}"
    return paths.model_inputs_root / tag / "samples"


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        fail(f"Missing JSON file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        fail(f"Expected JSON object at {path}")
    return data


def compute_file_sha256(path: Path) -> str:
    if not path.exists():
        fail(f"Cannot hash missing file: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_artifacts_or_fail(fingerprint: Dict[str, Any]) -> Dict[str, Any]:
    artifacts = fingerprint.get("artifacts")
    if not isinstance(artifacts, dict):
        fail("Fingerprint is missing `artifacts` hash section.")
    return artifacts


def _validate_split_index_hashes(
    paths: RunPaths,
    run_id: str | None,
    artifacts: Dict[str, Any],
    modalities: Sequence[str],
    splits: Sequence[str],
) -> None:
    expected_split_hashes = artifacts.get("split_index_sha256")
    if not isinstance(expected_split_hashes, dict):
        fail("Fingerprint missing `split_index_sha256`.")
    for modality in modalities:
        modality_map = expected_split_hashes.get(modality)
        if not isinstance(modality_map, dict):
            fail(f"Fingerprint missing split hash map for modality={modality}")
        for split in splits:
            expected = modality_map.get(split)
            if not isinstance(expected, str) or not expected:
                fail(f"Fingerprint missing split hash for {modality}/{split}")
            actual = compute_file_sha256(get_split_index_path(paths, run_id, modality, split))
            if actual != expected:
                fail(f"Split index hash mismatch: modality={modality}, split={split}. Rebuild dataloaders.")


def _validate_core_artifact_hashes(paths: RunPaths, run_id: str | None, artifacts: Dict[str, Any]) -> Dict[str, Any]:
    checks = (
        (get_normalization_stats_path(paths, run_id), "normalization_stats_sha256", "normalization_stats"),
        (get_learning_schema_path(paths, run_id), "learning_schema_sha256", "learning_schema"),
        (get_surrogate_targets_path(paths, run_id), "surrogate_targets_sha256", "surrogate_targets"),
        (get_shared_sample_manifest_path(paths, run_id), "shared_sample_manifest_sha256", "shared_sample_manifest"),
    )
    for path, artifact_key, label in checks:
        expected = str(artifacts.get(artifact_key, ""))
        if compute_file_sha256(path) != expected:
            fail(f"Artifact hash mismatch: {label}")

    manifest_payload = load_json(get_shared_sample_manifest_path(paths, run_id))
    cache_map = manifest_payload.get("cache_key_to_sha256")
    if not isinstance(cache_map, dict):
        fail("Invalid shared sample manifest schema.")
    expected_count = int(artifacts.get("shared_sample_count", -1))
    if expected_count != len(cache_map):
        fail(f"Shared sample count mismatch: fingerprint={expected_count}, manifest={len(cache_map)}")
    return manifest_payload


def _validate_shared_sample_hashes(sample_dir: Path, cache_map: Dict[str, Any], keys: Iterable[str]) -> None:
    missing: list[str] = []
    mismatch: list[str] = []
    for key in keys:
        skey = str(key)
        expected = cache_map.get(skey)
        sample_path = sample_dir / f"{skey}.pt"
        if not isinstance(expected, str) or not sample_path.exists():
            missing.append(skey)
        elif compute_file_sha256(sample_path) != expected:
            mismatch.append(skey)
        if len(missing) >= 10 or len(mismatch) >= 10:
            break
    if missing:
        fail(f"Missing shared sample entries/files: {missing}")
    if mismatch:
        fail(f"Shared sample hash mismatch keys: {mismatch}")


def validate_runtime_integrity(paths: RunPaths, run_id: str | None, modality: str, split: str) -> None:
    fingerprint = load_json(get_fingerprint_path(paths, run_id))
    artifacts = _get_artifacts_or_fail(fingerprint)
    modalities_to_validate = (modality, "fusion") if modality == "visual_only" else (modality,)
    _validate_split_index_hashes(paths, run_id, artifacts, modalities=modalities_to_validate, splits=(split,))
    manifest_payload = _validate_core_artifact_hashes(paths, run_id, artifacts)
    if modality not in {"visual_only", "fusion"}:
        return

    cache_map = manifest_payload.get("cache_key_to_sha256")
    sample_dir = Path(str(manifest_payload.get("sample_dir", "")))
    if not isinstance(cache_map, dict):
        fail("Invalid shared sample manifest cache map.")
    if not sample_dir.exists():
        build_cfg = _load_runtime_cache_context(paths, run_id)
        sample_dir = shared_cache_dir(paths, run_id, int(build_cfg["clip_len"]), int(build_cfg["img_size"]), str(build_cfg["cache_dtype"]))
    payload = load_pt_payload(get_split_index_path(paths, run_id, modality, split))
    keys = payload.get("cache_key")
    if not isinstance(keys, list):
        fail(f"Invalid {modality} index payload: missing cache_key list.")
    _validate_shared_sample_hashes(sample_dir=sample_dir, cache_map=cache_map, keys=[str(k) for k in keys])


def load_pt_payload(path: Path) -> Dict[str, Any]:
    if not path.exists():
        fail(f"Missing cache index file: {path}")
    payload = safe_torch_load(path)
    if not isinstance(payload, dict):
        fail(f"Expected dict payload in {path}")
    return payload


def safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_targets(payload: Dict[str, Any], target_mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    y_raw = payload.get("y_raw")
    y_z = payload.get("y_z")
    if not isinstance(y_raw, torch.Tensor) or not isinstance(y_z, torch.Tensor):
        fail("Invalid index payload: expected tensors `y_raw` and `y_z`.")
    if target_mode == "zscore":
        return y_z.float(), y_raw.float()
    if target_mode == "raw":
        return y_raw.float(), y_raw.float()
    fail(f"Unsupported target_mode: {target_mode}")
    return y_raw.float(), y_raw.float()


def _param_tensor(type_id: torch.Tensor, severity: torch.Tensor) -> torch.Tensor:
    return torch.stack((type_id.float(), severity.float()), dim=-1)


def _masked_severity(
    type_id: torch.Tensor,
    severity: torch.Tensor,
    is_ood_masked: torch.Tensor | None,
) -> torch.Tensor:
    mask = is_ood_masked.bool() if is_ood_masked is not None else type_id.long() == 0
    return torch.where(mask, torch.zeros_like(severity.float()), severity.float())


class ParamOnlyDataset(Dataset):
    def __init__(self, payload: Dict[str, Any], target_mode: str = "zscore"):
        y, y_raw = resolve_targets(payload, target_mode)
        type_id = payload.get("type_id")
        severity = payload.get("severity")
        masked = payload.get("is_ood_masked")
        if not isinstance(type_id, torch.Tensor):
            fail("ParamOnlyDataset payload missing `type_id` tensor.")
        if not isinstance(severity, torch.Tensor):
            fail("ParamOnlyDataset payload missing `severity` tensor.")
        if masked is not None and not isinstance(masked, torch.Tensor):
            fail("ParamOnlyDataset payload has invalid `is_ood_masked` tensor.")

        self.type_id = type_id.long()
        self.severity = severity.float()
        self.is_ood_masked = masked.long() if isinstance(masked, torch.Tensor) else None
        self.y = y
        self.y_raw = y_raw
        self.clip_id = [str(x) for x in payload.get("clip_id", [])]
        self.original_clip_id = [str(x) for x in payload.get("original_clip_id", [])]
        self.split = [str(x) for x in payload.get("split", [])]
        if len(self.type_id) != len(self.clip_id) or len(self.severity) != len(self.clip_id):
            fail("ParamOnlyDataset length mismatch.")

    def __len__(self) -> int:
        return len(self.clip_id)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        is_ood_masked = self.is_ood_masked[idx] if self.is_ood_masked is not None else None
        severity = _masked_severity(self.type_id[idx], self.severity[idx], is_ood_masked)
        item: Dict[str, Any] = {
            "type_id": self.type_id[idx],
            "severity": severity,
            "param": _param_tensor(self.type_id[idx], severity),
            "y": self.y[idx],
            "y_raw": self.y_raw[idx],
            "clip_id": self.clip_id[idx],
            "original_clip_id": self.original_clip_id[idx],
            "split": self.split[idx],
        }
        if is_ood_masked is not None:
            item["is_ood_masked"] = is_ood_masked
        return item


class VisualOnlyDataset(Dataset):
    def __init__(self, payload: Dict[str, Any], sample_dir: Path, target_mode: str = "zscore"):
        y, y_raw = resolve_targets(payload, target_mode)
        cache_key = payload.get("cache_key")
        if not isinstance(cache_key, list):
            fail("VisualOnlyDataset payload missing `cache_key` list.")
        self.cache_key = [str(x) for x in cache_key]
        self.sample_dir = sample_dir
        self.y = y
        self.y_raw = y_raw
        self.clip_id = [str(x) for x in payload.get("clip_id", [])]
        self.original_clip_id = [str(x) for x in payload.get("original_clip_id", [])]
        self.split = [str(x) for x in payload.get("split", [])]
        masked = payload.get("is_ood_masked")
        if masked is not None and not isinstance(masked, torch.Tensor):
            fail("VisualOnlyDataset payload has invalid `is_ood_masked` tensor.")
        self.is_ood_masked = masked.long() if isinstance(masked, torch.Tensor) else None
        if len(self.cache_key) != len(self.clip_id):
            fail("VisualOnlyDataset length mismatch.")
        if self.is_ood_masked is not None and len(self.is_ood_masked) != len(self.clip_id):
            fail("VisualOnlyDataset length mismatch(is_ood_masked).")

    def __len__(self) -> int:
        return len(self.clip_id)

    def _load_video(self, key: str) -> torch.Tensor:
        sample_path = self.sample_dir / f"{key}.pt"
        if not sample_path.exists():
            fail(f"Missing shared video sample: {sample_path}")
        payload = safe_torch_load(sample_path)
        if not isinstance(payload, dict) or "video" not in payload:
            fail(f"Invalid shared video payload: {sample_path}")
        video = payload["video"]
        if not isinstance(video, torch.Tensor):
            fail(f"Invalid video tensor in {sample_path}")
        return video.float().div(255.0) if video.dtype == torch.uint8 else video.float()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "video": self._load_video(self.cache_key[idx]),
            "y": self.y[idx],
            "y_raw": self.y_raw[idx],
            "clip_id": self.clip_id[idx],
            "original_clip_id": self.original_clip_id[idx],
            "split": self.split[idx],
        }
        if self.is_ood_masked is not None:
            item["is_ood_masked"] = self.is_ood_masked[idx]
        return item


class FusionDataset(VisualOnlyDataset):
    def __init__(self, payload: Dict[str, Any], sample_dir: Path, target_mode: str = "zscore"):
        super().__init__(payload=payload, sample_dir=sample_dir, target_mode=target_mode)
        type_id = payload.get("type_id")
        severity = payload.get("severity")
        masked = payload.get("is_ood_masked")
        if not isinstance(type_id, torch.Tensor):
            fail("FusionDataset payload missing `type_id` tensor.")
        if not isinstance(severity, torch.Tensor):
            fail("FusionDataset payload missing `severity` tensor.")
        if not isinstance(masked, torch.Tensor):
            fail("FusionDataset payload missing `is_ood_masked` tensor.")
        self.type_id = type_id.long()
        self.severity = severity.float()
        self.is_ood_masked = masked.long()
        if len(self.type_id) != len(self.clip_id) or len(self.severity) != len(self.clip_id):
            fail("FusionDataset length mismatch.")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base = super().__getitem__(idx)
        severity = _masked_severity(self.type_id[idx], self.severity[idx], self.is_ood_masked[idx])
        base["type_id"] = self.type_id[idx]
        base["severity"] = severity
        base["param"] = _param_tensor(self.type_id[idx], severity)
        base["is_ood_masked"] = self.is_ood_masked[idx]
        return base


def _load_runtime_cache_context(paths: RunPaths, run_id: str | None) -> Dict[str, Any]:
    fp = load_json(get_fingerprint_path(paths, run_id))
    build_cfg = fp.get("build_config")
    if not isinstance(build_cfg, dict):
        fail(f"Fingerprint missing build_config: {get_fingerprint_path(paths, run_id)}")
    return build_cfg


def _attach_visual_ood_mask_from_fusion(payload: Dict[str, Any], paths: RunPaths, run_id: str | None, split: str) -> Dict[str, Any]:
    """Backfill visual-only OOD metadata only for evaluation stratification.

    Visual models do not consume this key as input; it is attached so OOD
    subset metrics can be computed without rebuilding existing artifacts.
    """
    if "is_ood_masked" in payload:
        return payload
    fusion_path = get_split_index_path(paths, run_id, "fusion", split)
    if not fusion_path.exists():
        return payload
    fusion_payload = load_pt_payload(fusion_path)
    mask = fusion_payload.get("is_ood_masked")
    if not isinstance(mask, torch.Tensor):
        return payload
    visual_ids = [str(x) for x in payload.get("clip_id", [])]
    fusion_ids = [str(x) for x in fusion_payload.get("clip_id", [])]
    if visual_ids != fusion_ids:
        fail("Cannot backfill visual-only OOD mask: visual/fusion split clip_id order differs.")
    payload = dict(payload)
    payload["is_ood_masked"] = mask.long()
    return payload


def _build_single_dataloader(
    modality: str,
    split: str,
    paths: RunPaths,
    run_id: str | None,
    target_mode: str,
    batch_size: int,
    num_workers: int,
    runtime_build_cfg: Dict[str, Any] | None = None,
    shuffle: bool | None = None,
) -> DataLoader:
    if modality not in MODALITIES:
        fail(f"Unsupported modality: {modality}")
    if split not in SPLITS:
        fail(f"Invalid split: {split}. expected one of {SPLITS}")

    build_cfg = runtime_build_cfg if runtime_build_cfg is not None else _load_runtime_cache_context(paths, run_id)
    for key in ("img_size", "clip_len", "cache_dtype"):
        if key not in build_cfg:
            fail(f"runtime build config missing key: {key}")
    sample_dir = shared_cache_dir(paths, run_id, int(build_cfg["clip_len"]), int(build_cfg["img_size"]), str(build_cfg["cache_dtype"]))

    payload = load_pt_payload(get_split_index_path(paths, run_id, modality, split))
    if modality == "visual_only":
        payload = _attach_visual_ood_mask_from_fusion(payload, paths, run_id, split)
    if modality == "param_only":
        dataset: Dataset = ParamOnlyDataset(payload=payload, target_mode=target_mode)
    elif modality == "visual_only":
        dataset = VisualOnlyDataset(payload=payload, sample_dir=sample_dir, target_mode=target_mode)
    else:
        dataset = FusionDataset(payload=payload, sample_dir=sample_dir, target_mode=target_mode)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train") if shuffle is None else shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers > 0),
        drop_last=bool(split == "train" and modality in {"visual_only", "fusion"} and batch_size > 1 and len(dataset) > batch_size),
    )


def build_dataloaders(
    modality: str,
    split: str | Sequence[str],
    batch_size: int,
    num_workers: int,
    target_mode: str = "zscore",
    use_cache: bool = True,
    run_id: str | None = None,
    config_path: str | Path | None = None,
    paths_override: RunPaths | None = None,
    validate_integrity: bool = True,
    runtime_build_cfg: Dict[str, Any] | None = None,
    shuffle: bool | None = None,
) -> DataLoader | Dict[str, DataLoader]:
    if not use_cache:
        fail("Only use_cache=True is supported.")

    paths = paths_override if paths_override is not None else resolve_paths(config_path=config_path, run_id=run_id)
    if isinstance(split, str):
        if validate_integrity:
            validate_runtime_integrity(paths=paths, run_id=run_id, modality=modality, split=split)
        return _build_single_dataloader(
            modality=modality,
            split=split,
            paths=paths,
            run_id=run_id,
            target_mode=target_mode,
            batch_size=batch_size,
            num_workers=num_workers,
            runtime_build_cfg=runtime_build_cfg,
            shuffle=shuffle,
        )

    loaders: Dict[str, DataLoader] = {}
    for sp in split:
        split_name = str(sp)
        if validate_integrity:
            validate_runtime_integrity(paths=paths, run_id=run_id, modality=modality, split=split_name)
        loaders[split_name] = _build_single_dataloader(
            modality=modality,
            split=split_name,
            paths=paths,
            run_id=run_id,
            target_mode=target_mode,
            batch_size=batch_size,
            num_workers=num_workers,
            runtime_build_cfg=runtime_build_cfg,
            shuffle=shuffle if split_name == "train" else False,
        )
    return loaders


def load_normalization_stats(config_path: str | Path | None = None, run_id: str | None = None) -> Dict[str, Any]:
    paths = resolve_paths(config_path=config_path, run_id=run_id)
    return load_json(get_normalization_stats_path(paths, run_id))


def type_vocab_size(normalization_stats: Dict[str, Any]) -> int:
    mapping = normalization_stats.get("degradation_type_to_id", {})
    if not isinstance(mapping, dict) or not mapping:
        return 1
    return max(int(v) for v in mapping.values()) + 1
