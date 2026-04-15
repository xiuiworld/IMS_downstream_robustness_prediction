#!/usr/bin/env python3
"""
Proposal-compliant canonical dataloader builder.

Phase 2 bridge that converts interim manifests into strict training-ready inputs:
1) fail-fast master table build (clip_manifest + target_manifest)
2) proposal-strict OOD masking for holdout degradation types
3) train-only fit transforms (label z-score, type id map, param scaling)
4) modality-pure split indices (param_only / visual_only / fusion)
5) shared video cache + split index `.pt`
6) cache fingerprinting and provenance report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pandas is required. Install with `pip install pandas`.") from exc

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover
    raise SystemExit("torch is required. Install with `pip install torch`.") from exc

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pyyaml is required. Install with `pip install pyyaml`.") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS: Tuple[str, ...] = ("train", "val", "test")
SUPPORTED_FRAME_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SCHEMA_VERSION = "05_create_dataloaders.v3.proposal_compliant"
SHARED_CACHE_SCHEMA_VERSION = "shared_video_cache.v2"
DATASET_CONTRACT_VERSION = "dataset_contract.v2"
UNSET_PARAM_SENTINEL = -1.0
PROPOSAL_STRICT_TRAIN_ALLOWED_TYPES: Tuple[str, ...] = ("blur", "pixelate")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical dataloaders for Phase 2.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--run_id", type=str, default=None, help="None/canonical or e.g. smoke")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--clip_len", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--target_mode", choices=("zscore", "raw"), default="zscore")
    parser.add_argument("--cache_dtype", choices=("uint8", "float16", "float32"), default="uint8")
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--skip_sanity_check", action="store_true")
    parser.add_argument("--ood_policy", choices=("proposal_strict", "none"), default="proposal_strict")
    parser.add_argument(
        "--ood_holdout_types",
        type=str,
        default="h264_local",
        help="Comma-separated degradation types for strict OOD holdout.",
    )
    parser.add_argument(
        "--strict_train_allowed_types",
        type=str,
        default=",".join(PROPOSAL_STRICT_TRAIN_ALLOWED_TYPES),
        help="Comma-separated allowed degradation types in train for proposal_strict.",
    )
    parser.add_argument(
        "--allow_train_type_subset_for_smoke",
        action="store_true",
        help="Allow subset of strict train types only for smoke run_ids.",
    )
    return parser.parse_args()


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


def resolve_run_paths(config: Dict[str, Any], run_id: str | None) -> RunPaths:
    path_cfg = config.get("paths", {})
    if not isinstance(path_cfg, dict):
        fail("config.yaml missing `paths` mapping.")
    manifest_rel = path_cfg.get("manifest_dir", "data/interim/manifests")
    processed_rel = path_cfg.get("processed_dir", "data/processed")
    manifest_dir = PROJECT_ROOT / manifest_rel
    processed_root = PROJECT_ROOT / processed_rel
    suffix = run_suffix(run_id)

    clip_manifest = manifest_dir / "clip_manifest.csv"
    target_manifest = manifest_dir / with_run_suffix("target_manifest", suffix, ".csv")
    target_stats = manifest_dir / with_run_suffix("target_stats", suffix, ".csv")
    run_config_snapshot = manifest_dir / with_run_suffix("run_config_snapshot", suffix, ".json")

    targets_root = processed_root / "targets"
    model_inputs_root = processed_root / "model_inputs"
    evaluation_root = processed_root / "evaluation"
    return RunPaths(
        clip_manifest=clip_manifest,
        target_manifest=target_manifest,
        target_stats=target_stats,
        run_config_snapshot=run_config_snapshot,
        processed_root=processed_root,
        targets_root=targets_root,
        model_inputs_root=model_inputs_root,
        evaluation_root=evaluation_root,
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_required(path: Path, required_cols: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        fail(f"Required CSV missing: {path}")
    df = pd.read_csv(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        fail(f"{path} missing required columns: {missing}")
    return df


def normalize_param_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return ""
    try:
        v = float(text)
    except ValueError:
        return text
    if np.isnan(v):
        return ""
    if float(v).is_integer():
        return str(int(round(v)))
    normalized = f"{v:.12f}".rstrip("0").rstrip(".")
    return normalized if normalized != "" else "0"


def parse_float_strict(value: Any, field_name: str) -> float:
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        fail(f"Missing numeric value in `{field_name}`")
    try:
        parsed = float(text)
    except ValueError as exc:
        fail(f"Invalid float value in `{field_name}`: {value}")  # pragma: no cover
        raise exc
    if np.isnan(parsed):
        fail(f"NaN encountered in `{field_name}`")
    return parsed


def compute_file_sha256(path: Path) -> str:
    if not path.exists():
        fail(f"Cannot hash missing file: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_sample_key(clip_id: str, degradation_type: str, degradation_param_norm: str) -> str:
    token = f"{clip_id}|{degradation_type}|{degradation_param_norm}"
    return hashlib.sha1(token.encode("utf-8")).hexdigest()[:16]


def assert_unique(df: pd.DataFrame, key_cols: Sequence[str], tag: str) -> None:
    dup_mask = df.duplicated(list(key_cols), keep=False)
    if not dup_mask.any():
        return
    preview = df.loc[dup_mask, list(key_cols)].head(8).to_dict(orient="records")
    fail(f"{tag} has duplicate keys on {list(key_cols)}. sample={preview}")


def required_input_files(paths: RunPaths) -> List[Path]:
    return [paths.clip_manifest, paths.target_manifest, paths.target_stats, paths.run_config_snapshot]


def validate_required_inputs(paths: RunPaths) -> None:
    missing = [str(p) for p in required_input_files(paths) if not p.exists()]
    if missing:
        fail(f"Missing required inputs: {missing}")


def build_master_table(paths: RunPaths) -> pd.DataFrame:
    clip_df = read_csv_required(
        paths.clip_manifest,
        required_cols=(
            "clip_id",
            "sequence_name",
            "split",
            "start_frame",
            "end_frame",
            "degradation_type",
            "degradation_param",
            "file_path",
        ),
    )
    target_df = read_csv_required(
        paths.target_manifest,
        required_cols=(
            "clip_id",
            "original_clip_id",
            "degradation_type",
            "degradation_param",
            "p_orig_map",
            "p_orig_hota",
            "p_anon_map",
            "p_anon_hota",
            "delta_map",
            "delta_hota",
        ),
    )
    obf_df = clip_df.loc[clip_df["degradation_type"] != "original"].copy()
    if obf_df.empty:
        fail("clip_manifest has no obfuscated rows; cannot build surrogate training targets.")

    obf_df["original_clip_id"] = obf_df["clip_id"].astype(str)
    obf_df["degradation_param_norm"] = obf_df["degradation_param"].map(normalize_param_value)
    target_df["degradation_param_norm"] = target_df["degradation_param"].map(normalize_param_value)

    join_cols = ("original_clip_id", "degradation_type", "degradation_param_norm")
    assert_unique(obf_df, join_cols, "clip_manifest(obfuscated)")
    assert_unique(target_df, join_cols, "target_manifest")

    merged = obf_df.merge(
        target_df,
        on=list(join_cols),
        how="inner",
        validate="one_to_one",
        suffixes=("_clip", "_target"),
    )

    if len(merged) != len(target_df):
        missing = (
            target_df.merge(obf_df[list(join_cols)], on=list(join_cols), how="left", indicator=True)
            .loc[lambda d: d["_merge"] == "left_only", list(join_cols)]
            .head(10)
            .to_dict(orient="records")
        )
        fail(
            "Join mismatch: master row count differs from target_manifest "
            f"({len(merged)} vs {len(target_df)}). missing target keys sample={missing}"
        )

    merged["clip_id_final"] = merged["clip_id_target"].astype(str)
    merged["original_clip_id_final"] = merged["original_clip_id"].astype(str)
    merged["degradation_type_final"] = merged["degradation_type"].astype(str)
    merged["degradation_param_final"] = merged["degradation_param_target"]
    merged["degradation_param_value"] = merged["degradation_param_target"].map(
        lambda x: parse_float_strict(x, "degradation_param")
    )
    merged["delta_map_raw"] = merged["delta_map"].map(lambda x: parse_float_strict(x, "delta_map"))
    merged["delta_hota_raw"] = merged["delta_hota"].map(lambda x: parse_float_strict(x, "delta_hota"))
    merged["p_orig_map"] = merged["p_orig_map"].map(lambda x: parse_float_strict(x, "p_orig_map"))
    merged["p_orig_hota"] = merged["p_orig_hota"].map(lambda x: parse_float_strict(x, "p_orig_hota"))
    merged["p_anon_map"] = merged["p_anon_map"].map(lambda x: parse_float_strict(x, "p_anon_map"))
    merged["p_anon_hota"] = merged["p_anon_hota"].map(lambda x: parse_float_strict(x, "p_anon_hota"))

    master = merged[
        [
            "clip_id_final",
            "original_clip_id_final",
            "sequence_name",
            "split",
            "file_path",
            "start_frame",
            "end_frame",
            "degradation_type_final",
            "degradation_param_final",
            "degradation_param_norm",
            "degradation_param_value",
            "p_orig_map",
            "p_orig_hota",
            "p_anon_map",
            "p_anon_hota",
            "delta_map_raw",
            "delta_hota_raw",
        ]
    ].rename(
        columns={
            "clip_id_final": "clip_id",
            "original_clip_id_final": "original_clip_id",
            "degradation_type_final": "degradation_type",
            "degradation_param_final": "degradation_param",
        }
    )

    if master["split"].isna().any():
        fail("Missing `split` values in master table.")
    invalid_splits = sorted(set(master["split"].astype(str)) - set(SPLITS))
    if invalid_splits:
        fail(f"Unexpected split values in master: {invalid_splits}")
    if master[["clip_id", "original_clip_id", "file_path", "degradation_type"]].isna().any().any():
        fail("Null values found in required identifier columns after join.")

    abs_paths = master["file_path"].astype(str).map(lambda p: str((PROJECT_ROOT / p).resolve()))
    missing_mask = ~abs_paths.map(lambda p: Path(p).exists())
    if missing_mask.any():
        sample = master.loc[missing_mask, ["clip_id", "file_path"]].head(10).to_dict(orient="records")
        fail(f"Missing clip directories referenced by master table. sample={sample}")
    master["abs_file_path"] = abs_paths
    return master


def fit_and_apply_transforms(
    master: pd.DataFrame,
    ood_policy: str,
    holdout_types: Sequence[str],
    strict_train_allowed_types: Sequence[str],
    allow_train_type_subset: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    holdout_set = {t.strip() for t in holdout_types if t.strip()}
    train_df = master.loc[master["split"] == "train"].copy()
    if train_df.empty:
        fail("No train rows in master table; cannot fit train-only transforms.")

    if ood_policy == "proposal_strict":
        strict_allowed_set = {t.strip() for t in strict_train_allowed_types if t.strip()}
        if not strict_allowed_set:
            fail("proposal_strict requires non-empty strict_train_allowed_types.")
        if holdout_set & strict_allowed_set:
            fail(
                "Invalid configuration: holdout types overlap with strict train allowed types. "
                f"holdout={sorted(holdout_set)}, allowed={sorted(strict_allowed_set)}"
            )

        train_types_set = set(train_df["degradation_type"].astype(str).unique().tolist())
        violating_holdout = train_types_set & holdout_set
        if violating_holdout:
            sample = train_df.loc[train_df["degradation_type"].isin(violating_holdout)][
                ["clip_id", "degradation_type", "degradation_param", "split"]
            ].head(10)
            fail(
                "OOD protocol violation: holdout degradation type appears in train split. "
                f"holdout={sorted(holdout_set)}, sample={sample.to_dict(orient='records')}"
            )

        unexpected = sorted(train_types_set - strict_allowed_set)
        if unexpected:
            fail(
                "proposal_strict violation: train split contains disallowed degradation types. "
                f"allowed={sorted(strict_allowed_set)}, found_unexpected={unexpected}"
            )

        missing_required = sorted(strict_allowed_set - train_types_set)
        if missing_required and not allow_train_type_subset:
            fail(
                "proposal_strict violation: train split is missing required degradation types. "
                f"missing={missing_required}, found={sorted(train_types_set)}"
            )
        if not train_types_set:
            fail("proposal_strict violation: empty train degradation set.")
    else:
        strict_allowed_set = set()

    train_types = sorted(train_df["degradation_type"].astype(str).unique().tolist())
    type_to_id: Dict[str, int] = {"UNK": 0}
    for idx, deg_type in enumerate(train_types, start=1):
        type_to_id[deg_type] = idx

    param_ranges: Dict[str, Dict[str, float]] = {}
    for deg_type, g in train_df.groupby("degradation_type"):
        min_v = float(g["degradation_param_value"].min())
        max_v = float(g["degradation_param_value"].max())
        param_ranges[str(deg_type)] = {"min": min_v, "max": max_v}

    map_mean = float(train_df["delta_map_raw"].mean())
    hota_mean = float(train_df["delta_hota_raw"].mean())
    map_std = float(train_df["delta_map_raw"].std(ddof=0))
    hota_std = float(train_df["delta_hota_raw"].std(ddof=0))
    map_std = map_std if map_std > 0 else 1.0
    hota_std = hota_std if hota_std > 0 else 1.0

    def apply_row(row: pd.Series) -> Tuple[int, float, int]:
        deg_type = str(row["degradation_type"])
        split = str(row["split"])
        value = float(row["degradation_param_value"])
        is_holdout_eval = (
            ood_policy == "proposal_strict"
            and deg_type in holdout_set
            and split in {"val", "test"}
        )
        if is_holdout_eval:
            return 0, UNSET_PARAM_SENTINEL, 1

        if deg_type not in type_to_id:
            fail(
                "Unknown degradation type outside strict holdout masking. "
                f"type={deg_type}, split={split}, clip_id={row['clip_id']}"
            )
        bounds = param_ranges.get(deg_type)
        if bounds is None:
            fail(f"Missing param range for type={deg_type}")
        lo, hi = bounds["min"], bounds["max"]
        if hi == lo:
            scaled = 0.0
        else:
            scaled = (value - lo) / (hi - lo)
            scaled = float(np.clip(scaled, 0.0, 1.0))
        return int(type_to_id[deg_type]), float(scaled), 0

    transformed = master.copy()
    mapped = transformed.apply(apply_row, axis=1, result_type="expand")
    transformed["degradation_type_id"] = mapped[0].astype(int)
    transformed["degradation_param_scaled"] = mapped[1].astype(float)
    transformed["type_id"] = transformed["degradation_type_id"].astype(int)
    transformed["severity"] = transformed["degradation_param_scaled"].astype(float)
    transformed["is_ood_masked"] = mapped[2].astype(int)

    transformed["delta_map_z"] = (transformed["delta_map_raw"] - map_mean) / map_std
    transformed["delta_hota_z"] = (transformed["delta_hota_raw"] - hota_mean) / hota_std

    transform_meta: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "fit_split": "train",
        "ood_policy": ood_policy,
        "ood_holdout_types": sorted(holdout_set),
        "target_mode_default": "zscore",
        "label_stats": {
            "delta_map": {"mean": map_mean, "std": map_std},
            "delta_hota": {"mean": hota_mean, "std": hota_std},
        },
        "degradation_type_to_id": type_to_id,
        "degradation_param_range": param_ranges,
        "strict_train_allowed_types": sorted(strict_allowed_set),
        "allow_train_type_subset": bool(allow_train_type_subset),
    }
    return transformed, transform_meta


def split_filename(split: str, run_id: str | None, ext: str) -> str:
    suffix = run_suffix(run_id)
    if suffix == "":
        return f"{split}{ext}"
    return f"{split}{suffix}{ext}"


def get_surrogate_targets_path(paths: RunPaths, run_id: str | None) -> Path:
    suffix = run_suffix(run_id)
    return paths.targets_root / with_run_suffix("surrogate_targets", suffix, ".csv")


def get_normalization_stats_path(paths: RunPaths, run_id: str | None) -> Path:
    suffix = run_suffix(run_id)
    return paths.model_inputs_root / "fusion" / with_run_suffix("normalization_stats", suffix, ".json")


def get_learning_schema_path(paths: RunPaths, run_id: str | None) -> Path:
    suffix = run_suffix(run_id)
    return paths.model_inputs_root / with_run_suffix("learning_schema", suffix, ".json")


def get_shared_sample_manifest_path(paths: RunPaths, run_id: str | None) -> Path:
    suffix = run_suffix(run_id)
    return paths.model_inputs_root / with_run_suffix("shared_sample_manifest", suffix, ".json")


def get_split_index_path(paths: RunPaths, run_id: str | None, modality: str, split: str) -> Path:
    return paths.model_inputs_root / modality / split_filename(split, run_id, ".pt")


def collect_split_index_hashes(paths: RunPaths, run_id: str | None) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for modality in ("param_only", "visual_only", "fusion"):
        out[modality] = {}
        for split in SPLITS:
            idx_path = get_split_index_path(paths, run_id, modality, split)
            out[modality][split] = compute_file_sha256(idx_path)
    return out


def build_shared_sample_manifest(
    paths: RunPaths,
    run_id: str | None,
    sample_dir: Path,
    cache_keys: Sequence[str],
) -> Dict[str, Any]:
    key_to_sha: Dict[str, str] = {}
    for key in sorted(set(str(k) for k in cache_keys)):
        sample_path = sample_dir / f"{key}.pt"
        if not sample_path.exists():
            fail(f"Missing shared sample for manifest build: {sample_path}")
        key_to_sha[key] = compute_file_sha256(sample_path)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "sample_dir": str(sample_dir),
        "sample_count": len(key_to_sha),
        "cache_key_to_sha256": key_to_sha,
    }
    manifest_path = get_shared_sample_manifest_path(paths, run_id)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return {
        "manifest_path": str(manifest_path),
        "sample_count": int(len(key_to_sha)),
        "sample_dir": str(sample_dir),
    }


def modality_columns(modality: str) -> List[str]:
    base_labels = ["delta_map_raw", "delta_hota_raw", "delta_map_z", "delta_hota_z"]
    base_meta = ["clip_id", "original_clip_id", "sequence_name", "split"]
    if modality == "param_only":
        return base_meta + [
            "type_id",
            "severity",
            "is_ood_masked",
            *base_labels,
        ]
    if modality == "visual_only":
        return base_meta + ["file_path", "start_frame", "end_frame", "cache_key", *base_labels]
    if modality == "fusion":
        return base_meta + [
            "file_path",
            "start_frame",
            "end_frame",
            "cache_key",
            "type_id",
            "severity",
            "is_ood_masked",
            *base_labels,
        ]
    fail(f"Unsupported modality: {modality}")
    return []


def write_split_csvs(
    master: pd.DataFrame,
    paths: RunPaths,
    run_id: str | None,
    transform_meta: Dict[str, Any],
) -> Dict[str, Any]:
    ensure_dir(paths.targets_root)
    ensure_dir(paths.model_inputs_root)
    for modality in ("param_only", "visual_only", "fusion"):
        ensure_dir(paths.model_inputs_root / modality)

    suffix = run_suffix(run_id)
    master_path = paths.targets_root / with_run_suffix("surrogate_targets", suffix, ".csv")
    master.to_csv(master_path, index=False)

    split_counts: Dict[str, Dict[str, int]] = {m: {} for m in ("param_only", "visual_only", "fusion")}
    for modality in ("param_only", "visual_only", "fusion"):
        cols = modality_columns(modality)
        mdir = paths.model_inputs_root / modality
        for split in SPLITS:
            sdf = master.loc[master["split"] == split, cols].copy()
            csv_path = mdir / split_filename(split, run_id, ".csv")
            sdf.to_csv(csv_path, index=False)
            split_counts[modality][split] = int(len(sdf))

    norm_path = paths.model_inputs_root / "fusion" / with_run_suffix("normalization_stats", suffix, ".json")
    norm_payload = dict(transform_meta)
    norm_payload["row_counts"] = {
        split: int((master["split"] == split).sum()) for split in SPLITS
    }
    norm_payload["masked_counts"] = {
        split: int(((master["split"] == split) & (master["is_ood_masked"] == 1)).sum())
        for split in SPLITS
    }
    with norm_path.open("w", encoding="utf-8") as f:
        json.dump(norm_payload, f, indent=2, ensure_ascii=False)

    schema_path = paths.model_inputs_root / with_run_suffix("learning_schema", suffix, ".json")
    schema_payload = {
        "schema_version": DATASET_CONTRACT_VERSION,
        "modalities": {
            "param_only": {
                "csv_columns": modality_columns("param_only"),
                "dataset_keys": [
                    "type_id",
                    "severity",
                    "is_ood_masked",
                    "y",
                    "y_raw",
                    "clip_id",
                    "original_clip_id",
                    "split",
                ],
            },
            "visual_only": {
                "csv_columns": modality_columns("visual_only"),
                "dataset_keys": ["video", "y", "y_raw", "clip_id", "original_clip_id", "split"],
            },
            "fusion": {
                "csv_columns": modality_columns("fusion"),
                "dataset_keys": [
                    "video",
                    "type_id",
                    "severity",
                    "is_ood_masked",
                    "y",
                    "y_raw",
                    "clip_id",
                    "original_clip_id",
                    "split",
                ],
            },
        },
    }
    with schema_path.open("w", encoding="utf-8") as f:
        json.dump(schema_payload, f, indent=2, ensure_ascii=False)

    return {
        "master_path": str(master_path),
        "normalization_stats_path": str(norm_path),
        "learning_schema_path": str(schema_path),
        "split_counts": split_counts,
    }


def fingerprint_payload(
    paths: RunPaths,
    args: argparse.Namespace,
    run_id: str | None,
) -> Dict[str, Any]:
    suffix = run_suffix(run_id)
    script_path = Path(__file__).resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "shared_cache_schema_version": SHARED_CACHE_SCHEMA_VERSION,
        "dataset_contract_version": DATASET_CONTRACT_VERSION,
        "run_suffix": suffix,
        "run_id": run_id if run_id is not None else "canonical",
        "builder": {
            "script_path": str(script_path),
            "script_sha256": compute_file_sha256(script_path),
        },
        "input_files": {
            "clip_manifest": {
                "path": str(paths.clip_manifest),
                "sha256": compute_file_sha256(paths.clip_manifest),
            },
            "target_manifest": {
                "path": str(paths.target_manifest),
                "sha256": compute_file_sha256(paths.target_manifest),
            },
            "target_stats": {
                "path": str(paths.target_stats),
                "sha256": compute_file_sha256(paths.target_stats),
            },
            "run_config_snapshot": {
                "path": str(paths.run_config_snapshot),
                "sha256": compute_file_sha256(paths.run_config_snapshot),
            },
        },
        "build_config": {
            "img_size": int(args.img_size),
            "clip_len": int(args.clip_len),
            "ood_policy": str(args.ood_policy),
            "ood_holdout_types": sorted(
                [t.strip() for t in str(args.ood_holdout_types).split(",") if t.strip()]
            ),
            "strict_train_allowed_types": sorted(
                [t.strip() for t in str(args.strict_train_allowed_types).split(",") if t.strip()]
            ),
            "allow_train_type_subset_for_smoke": bool(args.allow_train_type_subset_for_smoke),
            "target_mode": str(args.target_mode),
            "cache_dtype": str(args.cache_dtype),
        },
    }


def build_final_fingerprint(
    base_fingerprint: Dict[str, Any],
    paths: RunPaths,
    run_id: str | None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    normalization_stats_path = get_normalization_stats_path(paths, run_id)
    learning_schema_path = get_learning_schema_path(paths, run_id)
    surrogate_targets_path = get_surrogate_targets_path(paths, run_id)
    shared_manifest_path = get_shared_sample_manifest_path(paths, run_id)
    shared_manifest = load_json(shared_manifest_path)
    cache_map = shared_manifest.get("cache_key_to_sha256")
    if not isinstance(cache_map, dict):
        fail(f"Invalid shared sample manifest: {shared_manifest_path}")
    artifact_hashes: Dict[str, Any] = {
        "normalization_stats_sha256": compute_file_sha256(normalization_stats_path),
        "learning_schema_sha256": compute_file_sha256(learning_schema_path),
        "surrogate_targets_sha256": compute_file_sha256(surrogate_targets_path),
        "split_index_sha256": collect_split_index_hashes(paths, run_id),
        "shared_sample_manifest_sha256": compute_file_sha256(shared_manifest_path),
        "shared_sample_count": int(len(cache_map)),
    }
    final_fingerprint = json.loads(json.dumps(base_fingerprint))
    final_fingerprint["artifacts"] = artifact_hashes
    return final_fingerprint, artifact_hashes


def get_fingerprint_path(paths: RunPaths, run_id: str | None) -> Path:
    ensure_dir(paths.model_inputs_root)
    suffix = run_suffix(run_id)
    return paths.model_inputs_root / with_run_suffix("cache_fingerprint", suffix, ".json")


def get_partial_fingerprint_path(paths: RunPaths, run_id: str | None) -> Path:
    ensure_dir(paths.model_inputs_root)
    suffix = run_suffix(run_id)
    return paths.model_inputs_root / with_run_suffix("cache_fingerprint.partial", suffix, ".json")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        fail(f"Expected JSON object at {path}")
    return data


def _collect_rebuild_targets(
    paths: RunPaths,
    run_id: str | None,
    clip_len: int,
    img_size: int,
    cache_dtype: str,
) -> Tuple[List[Path], List[Path]]:
    suffix = run_suffix(run_id)
    files: List[Path] = [
        get_surrogate_targets_path(paths, run_id),
        get_learning_schema_path(paths, run_id),
        paths.model_inputs_root / with_run_suffix("cache_fingerprint", suffix, ".json"),
        paths.model_inputs_root / with_run_suffix("cache_fingerprint.partial", suffix, ".json"),
        get_normalization_stats_path(paths, run_id),
        get_shared_sample_manifest_path(paths, run_id),
        paths.evaluation_root / with_run_suffix("dataloader_build_report", suffix, ".json"),
        paths.evaluation_root / with_run_suffix("target_stats_source", suffix, ".csv"),
        paths.evaluation_root / with_run_suffix("run_config_snapshot_source", suffix, ".json"),
    ]
    for modality in ("param_only", "visual_only", "fusion"):
        mdir = paths.model_inputs_root / modality
        for split in SPLITS:
            files.append(mdir / split_filename(split, run_id, ".csv"))
            files.append(mdir / split_filename(split, run_id, ".pt"))
    dirs: List[Path] = [
        shared_cache_dir(
            paths=paths,
            run_id=run_id,
            clip_len=clip_len,
            img_size=img_size,
            cache_dtype=cache_dtype,
        ).parent
    ]
    return files, dirs


def cleanup_run_outputs_for_rebuild(
    paths: RunPaths,
    run_id: str | None,
    clip_len: int,
    img_size: int,
    cache_dtype: str,
) -> Dict[str, int]:
    files, dirs = _collect_rebuild_targets(paths, run_id, clip_len, img_size, cache_dtype)
    removed_files = 0
    removed_dirs = 0
    for f in files:
        if f.exists():
            f.unlink()
            removed_files += 1
    for d in dirs:
        if d.exists():
            shutil.rmtree(d)
            removed_dirs += 1
    return {"removed_files": removed_files, "removed_dirs": removed_dirs}


def _load_fingerprint_or_fail(paths: RunPaths, run_id: str | None) -> Dict[str, Any]:
    fp_path = get_fingerprint_path(paths, run_id)
    if not fp_path.exists():
        fail(f"Missing fingerprint. Build cache first: {fp_path}")
    return load_json(fp_path)


def _get_artifacts_or_fail(fingerprint: Dict[str, Any]) -> Dict[str, Any]:
    artifacts = fingerprint.get("artifacts")
    if not isinstance(artifacts, dict):
        fail("Fingerprint is missing `artifacts` hash section.")
    return artifacts


def _validate_core_artifact_hashes(
    paths: RunPaths,
    run_id: str | None,
    artifacts: Dict[str, Any],
) -> Dict[str, Any]:
    norm_path = get_normalization_stats_path(paths, run_id)
    schema_path = get_learning_schema_path(paths, run_id)
    surrogate_path = get_surrogate_targets_path(paths, run_id)
    manifest_path = get_shared_sample_manifest_path(paths, run_id)

    actual_norm = compute_file_sha256(norm_path)
    actual_schema = compute_file_sha256(schema_path)
    actual_surrogate = compute_file_sha256(surrogate_path)
    actual_manifest = compute_file_sha256(manifest_path)

    if actual_norm != str(artifacts.get("normalization_stats_sha256", "")):
        fail("Artifact hash mismatch: normalization_stats")
    if actual_schema != str(artifacts.get("learning_schema_sha256", "")):
        fail("Artifact hash mismatch: learning_schema")
    if actual_surrogate != str(artifacts.get("surrogate_targets_sha256", "")):
        fail("Artifact hash mismatch: surrogate_targets")
    if actual_manifest != str(artifacts.get("shared_sample_manifest_sha256", "")):
        fail("Artifact hash mismatch: shared_sample_manifest")

    manifest_payload = load_json(manifest_path)
    cache_map = manifest_payload.get("cache_key_to_sha256")
    if not isinstance(cache_map, dict):
        fail(f"Invalid shared sample manifest schema: {manifest_path}")
    expected_count = int(artifacts.get("shared_sample_count", -1))
    if expected_count != len(cache_map):
        fail(
            "Shared sample count mismatch. "
            f"fingerprint={expected_count}, manifest={len(cache_map)}"
        )
    manifest_count = int(manifest_payload.get("sample_count", -1))
    if manifest_count != len(cache_map):
        fail(
            "Shared sample manifest internal count mismatch. "
            f"sample_count={manifest_count}, map_size={len(cache_map)}"
        )
    return manifest_payload


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
            if not isinstance(expected, str) or expected == "":
                fail(f"Fingerprint missing split hash for {modality}/{split}")
            idx_path = get_split_index_path(paths, run_id, modality, split)
            actual = compute_file_sha256(idx_path)
            if actual != expected:
                fail(
                    f"Split index hash mismatch: modality={modality}, split={split}. "
                    "Use --rebuild_cache."
                )


def _validate_shared_sample_hashes(
    sample_dir: Path,
    cache_map: Dict[str, Any],
    keys: Iterable[str],
) -> None:
    missing: List[str] = []
    mismatch: List[str] = []
    for key in keys:
        skey = str(key)
        expected = cache_map.get(skey)
        if not isinstance(expected, str) or expected == "":
            missing.append(skey)
            if len(missing) >= 10:
                break
            continue
        sample_path = sample_dir / f"{skey}.pt"
        if not sample_path.exists():
            missing.append(str(sample_path))
            if len(missing) >= 10:
                break
            continue
        actual = compute_file_sha256(sample_path)
        if actual != expected:
            mismatch.append(skey)
            if len(mismatch) >= 10:
                break
    if missing:
        fail(f"Missing shared sample entries/files: {missing}")
    if mismatch:
        fail(f"Shared sample hash mismatch keys: {mismatch}")


def assert_cache_state_or_raise(
    paths: RunPaths,
    run_id: str | None,
    expected_base_fingerprint: Dict[str, Any],
    rebuild_cache: bool,
) -> bool:
    if rebuild_cache:
        return False

    fp_path = get_fingerprint_path(paths, run_id)
    if not fp_path.exists():
        return False

    existing = load_json(fp_path)
    existing_base = json.loads(json.dumps(existing))
    existing_base.pop("artifacts", None)
    if existing_base != expected_base_fingerprint:
        fail(
            "Cache fingerprint mismatch detected. Refusing stale cache reuse. "
            "Re-run with --rebuild_cache."
        )

    required: List[Path] = []
    suffix = run_suffix(run_id)
    required.append(get_surrogate_targets_path(paths, run_id))
    required.append(get_normalization_stats_path(paths, run_id))
    required.append(get_learning_schema_path(paths, run_id))
    required.append(get_shared_sample_manifest_path(paths, run_id))
    for modality in ("param_only", "visual_only", "fusion"):
        mdir = paths.model_inputs_root / modality
        for split in SPLITS:
            required.append(mdir / split_filename(split, run_id, ".csv"))
            required.append(mdir / split_filename(split, run_id, ".pt"))
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        fail(
            "Cache fingerprint matched but expected artifacts are missing. "
            f"Missing sample: {missing[:5]} (total={len(missing)}). "
            "Use --rebuild_cache."
        )

    artifacts = _get_artifacts_or_fail(existing)
    _validate_split_index_hashes(
        paths=paths,
        run_id=run_id,
        artifacts=artifacts,
        modalities=("param_only", "visual_only", "fusion"),
        splits=SPLITS,
    )
    manifest_payload = _validate_core_artifact_hashes(paths=paths, run_id=run_id, artifacts=artifacts)
    sample_dir = Path(str(manifest_payload.get("sample_dir", "")))
    if not sample_dir.exists():
        fail(f"Shared sample_dir from manifest does not exist: {sample_dir}")
    cache_map = manifest_payload.get("cache_key_to_sha256")
    if not isinstance(cache_map, dict):
        fail("Invalid shared sample manifest cache map.")
    unique_keys: set[str] = set()
    for split in SPLITS:
        visual_idx = get_split_index_path(paths, run_id, "visual_only", split)
        payload = load_pt_payload(visual_idx)
        keys = payload.get("cache_key")
        if not isinstance(keys, list):
            fail(f"Invalid visual index payload (cache_key list missing): {visual_idx}")
        for key in keys:
            unique_keys.add(str(key))
    _validate_shared_sample_hashes(sample_dir=sample_dir, cache_map=cache_map, keys=sorted(unique_keys))
    return True


def validate_runtime_integrity(
    paths: RunPaths,
    run_id: str | None,
    modality: str,
    split: str,
) -> None:
    fingerprint = _load_fingerprint_or_fail(paths, run_id)
    artifacts = _get_artifacts_or_fail(fingerprint)
    _validate_split_index_hashes(
        paths=paths,
        run_id=run_id,
        artifacts=artifacts,
        modalities=(modality,),
        splits=(split,),
    )
    manifest_payload = _validate_core_artifact_hashes(paths=paths, run_id=run_id, artifacts=artifacts)
    if modality not in {"visual_only", "fusion"}:
        return
    cache_map = manifest_payload.get("cache_key_to_sha256")
    if not isinstance(cache_map, dict):
        fail("Invalid shared sample manifest cache map.")
    sample_dir = Path(str(manifest_payload.get("sample_dir", "")))
    if not sample_dir.exists():
        fail(f"Shared sample_dir from manifest does not exist: {sample_dir}")
    idx_path = get_split_index_path(paths, run_id, modality, split)
    payload = load_pt_payload(idx_path)
    keys = payload.get("cache_key")
    if not isinstance(keys, list):
        fail(f"Invalid {modality} index payload (cache_key list missing): {idx_path}")
    _validate_shared_sample_hashes(sample_dir=sample_dir, cache_map=cache_map, keys=[str(k) for k in keys])


def copy_provenance_and_write_report(
    paths: RunPaths,
    run_id: str | None,
    fingerprint: Dict[str, Any],
    report_extra: Dict[str, Any],
) -> Path:
    ensure_dir(paths.evaluation_root)
    suffix = run_suffix(run_id)

    target_stats_copy = paths.evaluation_root / with_run_suffix("target_stats_source", suffix, ".csv")
    snapshot_copy = paths.evaluation_root / with_run_suffix("run_config_snapshot_source", suffix, ".json")
    shutil.copy2(paths.target_stats, target_stats_copy)
    shutil.copy2(paths.run_config_snapshot, snapshot_copy)

    report = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id if run_id is not None else "canonical",
        "input_fingerprint": fingerprint,
        "provenance_copies": {
            "target_stats_source": str(target_stats_copy),
            "run_config_snapshot_source": str(snapshot_copy),
        },
    }
    report.update(report_extra)

    report_path = paths.evaluation_root / with_run_suffix("dataloader_build_report", suffix, ".json")
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report_path


def shared_cache_dir(paths: RunPaths, run_id: str | None, clip_len: int, img_size: int, cache_dtype: str) -> Path:
    suffix = run_suffix(run_id)
    tag = f"shared_video_cache{suffix}_T{clip_len}_S{img_size}_{cache_dtype}"
    return paths.model_inputs_root / tag / "samples"


def detect_frame_extension(clip_dir: Path) -> str:
    if not clip_dir.exists():
        fail(f"Clip directory does not exist: {clip_dir}")
    files = [p for p in clip_dir.iterdir() if p.is_file()]
    if not files:
        fail(f"No frame files found in {clip_dir}")
    counts: Dict[str, int] = {}
    for f in files:
        ext = f.suffix.lower()
        if ext in SUPPORTED_FRAME_EXTS:
            counts[ext] = counts.get(ext, 0) + 1
    if not counts:
        fail(f"No supported frame extensions found in {clip_dir}; supported={SUPPORTED_FRAME_EXTS}")
    sorted_counts = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top_ext, top_count = sorted_counts[0]
    tied = [ext for ext, c in sorted_counts if c == top_count]
    if len(tied) > 1:
        fail(f"Ambiguous frame extension in {clip_dir}. candidates={sorted(counts.items())}")
    return top_ext


def _extract_frame_index(path: Path) -> int | None:
    match = re.search(r"(\d+)$", path.stem)
    if not match:
        return None
    return int(match.group(1))


def ordered_frame_paths(
    clip_dir: Path,
    clip_len: int,
    expected_start_frame: int | None = None,
    expected_end_frame: int | None = None,
) -> List[Path]:
    ext = detect_frame_extension(clip_dir)
    frames = sorted(clip_dir.glob(f"*{ext}"))
    if len(frames) != clip_len:
        fail(
            f"Frame count mismatch in {clip_dir}. expected_exact={clip_len}, found={len(frames)}"
        )

    parsed = [_extract_frame_index(p) for p in frames]
    if all(v is not None for v in parsed):
        frame_idx = [int(v) for v in parsed if v is not None]
        if len(set(frame_idx)) != len(frame_idx):
            fail(f"Duplicate frame indices detected in {clip_dir}")
        ordered = sorted(frame_idx)
        expected_seq = list(range(ordered[0], ordered[-1] + 1))
        if ordered != expected_seq:
            fail(f"Non-contiguous frame index sequence in {clip_dir}")
        if expected_start_frame is not None and ordered[0] != int(expected_start_frame):
            fail(
                f"Start frame mismatch in {clip_dir}. "
                f"expected={expected_start_frame}, found={ordered[0]}"
            )
        if expected_end_frame is not None and ordered[-1] != int(expected_end_frame):
            fail(
                f"End frame mismatch in {clip_dir}. "
                f"expected={expected_end_frame}, found={ordered[-1]}"
            )
        if len(frame_idx) != clip_len:
            fail(f"Frame index length mismatch in {clip_dir}")
    return frames


def read_frame_rgb(path: Path, img_size: int) -> np.ndarray:
    if cv2 is not None:
        frame_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            fail(f"Failed to read frame: {path}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        return frame_rgb

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Either opencv-python or pillow is required.") from exc
    with Image.open(path) as im:
        im = im.convert("RGB").resize((img_size, img_size))
        arr = np.array(im)
    return arr


def load_clip_video_tensor(
    clip_dir: Path,
    clip_len: int,
    img_size: int,
    cache_dtype: str,
    expected_start_frame: int | None = None,
    expected_end_frame: int | None = None,
) -> torch.Tensor:
    frames = ordered_frame_paths(
        clip_dir=clip_dir,
        clip_len=clip_len,
        expected_start_frame=expected_start_frame,
        expected_end_frame=expected_end_frame,
    )
    stacked = np.stack([read_frame_rgb(p, img_size) for p in frames], axis=0)  # [T, H, W, C]
    video = np.transpose(stacked, (0, 3, 1, 2))  # [T, C, H, W]

    if cache_dtype == "uint8":
        arr = video.astype(np.uint8, copy=False)
        return torch.from_numpy(arr)
    if cache_dtype == "float16":
        arr = (video.astype(np.float32) / 255.0).astype(np.float16)
        return torch.from_numpy(arr)
    if cache_dtype == "float32":
        arr = (video.astype(np.float32) / 255.0).astype(np.float32)
        return torch.from_numpy(arr)
    fail(f"Unsupported cache_dtype: {cache_dtype}")
    return torch.empty(0)


def build_shared_video_cache(
    master: pd.DataFrame,
    paths: RunPaths,
    run_id: str | None,
    clip_len: int,
    img_size: int,
    cache_dtype: str,
    rebuild_cache: bool,
) -> Dict[str, Any]:
    sample_dir = shared_cache_dir(paths, run_id, clip_len, img_size, cache_dtype)
    ensure_dir(sample_dir)

    assert_unique(master, ("cache_key",), "master(cache_key)")
    total = len(master)
    reused = 0
    built = 0

    for row in master.itertuples(index=False):
        cache_key = str(row.cache_key)
        clip_dir = Path(str(row.abs_file_path))
        sample_path = sample_dir / f"{cache_key}.pt"
        if sample_path.exists() and not rebuild_cache:
            reused += 1
            continue
        video_tensor = load_clip_video_tensor(
            clip_dir=clip_dir,
            clip_len=clip_len,
            img_size=img_size,
            cache_dtype=cache_dtype,
            expected_start_frame=int(row.start_frame),
            expected_end_frame=int(row.end_frame),
        )
        payload = {
            "video": video_tensor,
            "clip_id": str(row.clip_id),
            "original_clip_id": str(row.original_clip_id),
            "split": str(row.split),
            "cache_key": cache_key,
            "cache_dtype": cache_dtype,
            "clip_len": int(clip_len),
            "img_size": int(img_size),
        }
        torch.save(payload, sample_path)
        built += 1

    meta = {
        "sample_dir": str(sample_dir),
        "total_samples": int(total),
        "built_samples": int(built),
        "reused_samples": int(reused),
        "cache_dtype": cache_dtype,
        "clip_len": int(clip_len),
        "img_size": int(img_size),
    }
    return meta


def build_split_pt_indices(master: pd.DataFrame, paths: RunPaths, run_id: str | None) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {m: {} for m in ("param_only", "visual_only", "fusion")}
    for split in SPLITS:
        sdf = master.loc[master["split"] == split].copy()
        y_z = torch.tensor(sdf[["delta_map_z", "delta_hota_z"]].to_numpy(np.float32), dtype=torch.float32)
        y_raw = torch.tensor(
            sdf[["delta_map_raw", "delta_hota_raw"]].to_numpy(np.float32), dtype=torch.float32
        )
        type_id = torch.tensor(sdf["type_id"].to_numpy(np.int64), dtype=torch.long)
        severity = torch.tensor(sdf["severity"].to_numpy(np.float32), dtype=torch.float32)
        meta_common = {
            "clip_id": sdf["clip_id"].astype(str).tolist(),
            "original_clip_id": sdf["original_clip_id"].astype(str).tolist(),
            "split": sdf["split"].astype(str).tolist(),
            "y_z": y_z,
            "y_raw": y_raw,
        }

        param_payload = dict(meta_common)
        param_payload["type_id"] = type_id
        param_payload["severity"] = severity
        param_payload["is_ood_masked"] = torch.tensor(
            sdf["is_ood_masked"].to_numpy(np.int64), dtype=torch.long
        )
        torch.save(
            param_payload,
            paths.model_inputs_root / "param_only" / split_filename(split, run_id, ".pt"),
        )
        counts["param_only"][split] = int(len(sdf))

        visual_payload = dict(meta_common)
        visual_payload["cache_key"] = sdf["cache_key"].astype(str).tolist()
        torch.save(
            visual_payload,
            paths.model_inputs_root / "visual_only" / split_filename(split, run_id, ".pt"),
        )
        counts["visual_only"][split] = int(len(sdf))

        fusion_payload = dict(meta_common)
        fusion_payload["cache_key"] = sdf["cache_key"].astype(str).tolist()
        fusion_payload["type_id"] = type_id
        fusion_payload["severity"] = severity
        fusion_payload["is_ood_masked"] = torch.tensor(
            sdf["is_ood_masked"].to_numpy(np.int64), dtype=torch.long
        )
        torch.save(
            fusion_payload,
            paths.model_inputs_root / "fusion" / split_filename(split, run_id, ".pt"),
        )
        counts["fusion"][split] = int(len(sdf))
    return counts


def load_pt_payload(path: Path) -> Dict[str, Any]:
    if not path.exists():
        fail(f"Missing cache index file: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        fail(f"Expected dict payload in {path}")
    return payload


def resolve_targets(payload: Dict[str, Any], target_mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    y_raw = payload.get("y_raw")
    y_z = payload.get("y_z")
    if not isinstance(y_raw, torch.Tensor) or not isinstance(y_z, torch.Tensor):
        fail("Invalid index payload: expected tensors `y_raw` and `y_z`.")
    if target_mode == "zscore":
        return y_z, y_raw
    if target_mode == "raw":
        return y_raw, y_raw
    fail(f"Unsupported target_mode: {target_mode}")
    return y_raw, y_raw


class ParamOnlyDataset(Dataset):
    def __init__(self, payload: Dict[str, Any], target_mode: str = "zscore"):
        y, y_raw = resolve_targets(payload, target_mode)
        type_id = payload.get("type_id")
        severity = payload.get("severity")
        if not isinstance(type_id, torch.Tensor):
            fail("ParamOnlyDataset payload missing `type_id` tensor.")
        if not isinstance(severity, torch.Tensor):
            fail("ParamOnlyDataset payload missing `severity` tensor.")
        self.type_id = type_id.long()
        self.severity = severity.float()
        masked = payload.get("is_ood_masked")
        if masked is not None and not isinstance(masked, torch.Tensor):
            fail("ParamOnlyDataset payload has invalid `is_ood_masked` tensor.")
        self.is_ood_masked = masked.long() if isinstance(masked, torch.Tensor) else None
        self.y = y.float()
        self.y_raw = y_raw.float()
        self.clip_id = [str(x) for x in payload.get("clip_id", [])]
        self.original_clip_id = [str(x) for x in payload.get("original_clip_id", [])]
        self.split = [str(x) for x in payload.get("split", [])]
        if len(self.type_id) != len(self.clip_id):
            fail("ParamOnlyDataset length mismatch(type_id).")
        if len(self.severity) != len(self.clip_id):
            fail("ParamOnlyDataset length mismatch.")

    def __len__(self) -> int:
        return len(self.clip_id)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "type_id": self.type_id[idx],
            "severity": self.severity[idx],
            "y": self.y[idx],
            "y_raw": self.y_raw[idx],
            "clip_id": self.clip_id[idx],
            "original_clip_id": self.original_clip_id[idx],
            "split": self.split[idx],
            **(
                {"is_ood_masked": self.is_ood_masked[idx]}
                if self.is_ood_masked is not None
                else {}
            ),
        }


class VisualOnlyDataset(Dataset):
    def __init__(self, payload: Dict[str, Any], sample_dir: Path, target_mode: str = "zscore"):
        y, y_raw = resolve_targets(payload, target_mode)
        cache_key = payload.get("cache_key")
        if not isinstance(cache_key, list):
            fail("VisualOnlyDataset payload missing `cache_key` list.")
        self.cache_key = [str(x) for x in cache_key]
        self.sample_dir = sample_dir
        self.y = y.float()
        self.y_raw = y_raw.float()
        self.clip_id = [str(x) for x in payload.get("clip_id", [])]
        self.original_clip_id = [str(x) for x in payload.get("original_clip_id", [])]
        self.split = [str(x) for x in payload.get("split", [])]
        if len(self.cache_key) != len(self.clip_id):
            fail("VisualOnlyDataset length mismatch.")

    def __len__(self) -> int:
        return len(self.clip_id)

    def _load_video(self, key: str) -> torch.Tensor:
        sample_path = self.sample_dir / f"{key}.pt"
        if not sample_path.exists():
            fail(f"Missing shared video sample: {sample_path}")
        payload = torch.load(sample_path, map_location="cpu")
        if not isinstance(payload, dict) or "video" not in payload:
            fail(f"Invalid shared video payload: {sample_path}")
        video = payload["video"]
        if not isinstance(video, torch.Tensor):
            fail(f"Invalid video tensor in {sample_path}")
        if video.dtype == torch.uint8:
            return video.float().div(255.0)
        return video.float()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video = self._load_video(self.cache_key[idx])
        return {
            "video": video,
            "y": self.y[idx],
            "y_raw": self.y_raw[idx],
            "clip_id": self.clip_id[idx],
            "original_clip_id": self.original_clip_id[idx],
            "split": self.split[idx],
        }


class FusionDataset(VisualOnlyDataset):
    def __init__(self, payload: Dict[str, Any], sample_dir: Path, target_mode: str = "zscore"):
        super().__init__(payload=payload, sample_dir=sample_dir, target_mode=target_mode)
        type_id = payload.get("type_id")
        severity = payload.get("severity")
        if not isinstance(type_id, torch.Tensor):
            fail("FusionDataset payload missing `type_id` tensor.")
        if not isinstance(severity, torch.Tensor):
            fail("FusionDataset payload missing `severity` tensor.")
        masked = payload.get("is_ood_masked")
        if not isinstance(masked, torch.Tensor):
            fail("FusionDataset payload missing `is_ood_masked` tensor.")
        self.type_id = type_id.long()
        self.severity = severity.float()
        self.is_ood_masked = masked.long()
        if len(self.type_id) != len(self.clip_id):
            fail("FusionDataset length mismatch(type_id).")
        if len(self.severity) != len(self.clip_id):
            fail("FusionDataset length mismatch(severity).")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base = super().__getitem__(idx)
        base["type_id"] = self.type_id[idx]
        base["severity"] = self.severity[idx]
        base["is_ood_masked"] = self.is_ood_masked[idx]
        return base


def _load_runtime_cache_context(paths: RunPaths, run_id: str | None) -> Dict[str, Any]:
    fp_path = get_fingerprint_path(paths, run_id)
    if not fp_path.exists():
        fail(f"Missing fingerprint. Build cache first: {fp_path}")
    fp = load_json(fp_path)
    build_cfg = fp.get("build_config")
    if not isinstance(build_cfg, dict):
        fail(f"Fingerprint missing build_config: {fp_path}")
    return build_cfg


def _build_single_dataloader(
    modality: str,
    split: str,
    paths: RunPaths,
    run_id: str | None,
    target_mode: str,
    batch_size: int,
    num_workers: int,
    runtime_build_cfg: Dict[str, Any] | None = None,
) -> DataLoader:
    if split not in SPLITS:
        fail(f"Invalid split: {split}. expected one of {SPLITS}")

    build_cfg = runtime_build_cfg if runtime_build_cfg is not None else _load_runtime_cache_context(paths, run_id)
    if not isinstance(build_cfg, dict):
        fail("Invalid runtime_build_cfg; expected dict.")
    for key in ("img_size", "clip_len", "cache_dtype"):
        if key not in build_cfg:
            fail(f"runtime build config missing key: {key}")
    img_size = int(build_cfg["img_size"])
    clip_len = int(build_cfg["clip_len"])
    cache_dtype = str(build_cfg["cache_dtype"])
    sample_dir = shared_cache_dir(paths, run_id, clip_len, img_size, cache_dtype)

    index_path = paths.model_inputs_root / modality / split_filename(split, run_id, ".pt")
    payload = load_pt_payload(index_path)
    if modality == "param_only":
        dataset: Dataset = ParamOnlyDataset(payload=payload, target_mode=target_mode)
    elif modality == "visual_only":
        dataset = VisualOnlyDataset(payload=payload, sample_dir=sample_dir, target_mode=target_mode)
    elif modality == "fusion":
        dataset = FusionDataset(payload=payload, sample_dir=sample_dir, target_mode=target_mode)
    else:
        fail(f"Unsupported modality: {modality}")
        dataset = ParamOnlyDataset(payload=payload, target_mode=target_mode)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers > 0),
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
) -> DataLoader | Dict[str, DataLoader]:
    """
    Build DataLoader(s) from cached `.pt` artifacts.

    Contracts:
    - `param_only`: returns `type_id`, `severity`, `y`, `y_raw`, ids/split.
    - `visual_only`: returns `video`, `y`, `y_raw`, ids/split (no param).
    - `fusion`: returns `video`, `type_id`, `severity`, `y`, `y_raw`, `is_ood_masked`, ids/split.
    """
    if not use_cache:
        fail("Only use_cache=True is supported in this script.")

    if paths_override is not None:
        paths = paths_override
    else:
        if config_path is None:
            cfg_path = PROJECT_ROOT / "configs" / "config.yaml"
        else:
            cp = Path(config_path)
            cfg_path = cp if cp.is_absolute() else (PROJECT_ROOT / cp)
        config = load_yaml(cfg_path)
        paths = resolve_run_paths(config, run_id)
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
        )

    out: Dict[str, DataLoader] = {}
    for sp in split:
        if validate_integrity:
            validate_runtime_integrity(paths=paths, run_id=run_id, modality=modality, split=str(sp))
        out[str(sp)] = _build_single_dataloader(
            modality=modality,
            split=str(sp),
            paths=paths,
            run_id=run_id,
            target_mode=target_mode,
            batch_size=batch_size,
            num_workers=num_workers,
            runtime_build_cfg=runtime_build_cfg,
        )
    return out


def parse_holdout_types(text: str) -> List[str]:
    return [t.strip() for t in str(text).split(",") if t.strip()]


def parse_train_allowed_types(text: str) -> List[str]:
    return [t.strip() for t in str(text).split(",") if t.strip()]


def is_smoke_run(run_id: str | None) -> bool:
    if run_id is None:
        return False
    return "smoke" in run_id.lower()


def is_canonical_run(run_id: str | None) -> bool:
    if run_id is None:
        return True
    rid = str(run_id).strip().lower()
    return rid == "" or rid == "canonical"


def load_target_stats_effective_counts(
    path: Path,
) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]], bool]:
    df = read_csv_required(path, required_cols=("split", "included_targets"))
    reused_column_missing = "reused_completed_count" not in df.columns
    if reused_column_missing:
        df = df.copy()
        df["reused_completed_count"] = 0

    effective_counts: Dict[str, int] = {}
    raw_counts: Dict[str, Dict[str, int]] = {}
    for row in df.itertuples(index=False):
        split = str(row.split).strip()
        included = int(float(row.included_targets))
        reused = int(float(row.reused_completed_count))
        effective = included + reused
        effective_counts[split] = effective
        raw_counts[split] = {
            "included_targets": included,
            "reused_completed_count": reused,
            "effective_included": effective,
        }
    if "GLOBAL" not in effective_counts:
        fail(f"target_stats missing GLOBAL row: {path}")
    return effective_counts, raw_counts, reused_column_missing


def validate_master_against_target_stats(
    master: pd.DataFrame,
    target_stats_effective_counts: Dict[str, int],
    require_all_splits: bool,
) -> None:
    if require_all_splits:
        required_rows = list(SPLITS) + ["GLOBAL"]
        missing = [s for s in required_rows if s not in target_stats_effective_counts]
        if missing:
            fail(f"target_stats is missing required split rows: {missing}")
    for split in SPLITS:
        if split not in target_stats_effective_counts:
            continue
        expected = int(target_stats_effective_counts[split])
        actual = int((master["split"] == split).sum())
        if actual != expected:
            fail(
                f"target_stats mismatch for split={split}: "
                f"master={actual}, target_stats.effective_included={expected}"
            )
    expected_global = int(target_stats_effective_counts["GLOBAL"])
    actual_global = int(len(master))
    if actual_global != expected_global:
        fail(
            "target_stats mismatch for GLOBAL: "
            f"master={actual_global}, target_stats.effective_included={expected_global}"
        )


def quick_purity_assertions(paths: RunPaths, run_id: str | None) -> None:
    forbidden_visual = {"degradation_type", "degradation_param", "type_id", "severity"}
    forbidden_param = {"file_path", "cache_key", "degradation_type", "degradation_param"}
    forbidden_fusion = {"degradation_type", "degradation_param"}
    for split in SPLITS:
        vcsv = paths.model_inputs_root / "visual_only" / split_filename(split, run_id, ".csv")
        pcsv = paths.model_inputs_root / "param_only" / split_filename(split, run_id, ".csv")
        fcsv = paths.model_inputs_root / "fusion" / split_filename(split, run_id, ".csv")
        vdf = pd.read_csv(vcsv)
        pdf = pd.read_csv(pcsv)
        fdf = pd.read_csv(fcsv)
        if forbidden_visual.intersection(set(vdf.columns)):
            fail(f"visual_only purity violated in {vcsv}")
        if forbidden_param.intersection(set(pdf.columns)):
            fail(f"param_only purity violated in {pcsv}")
        if forbidden_fusion.intersection(set(fdf.columns)):
            fail(f"fusion purity violated in {fcsv}")


def run_sanity_batches(
    args: argparse.Namespace,
    run_id: str | None,
    paths: RunPaths,
) -> Dict[str, Dict[str, Any]]:
    checks: Dict[str, Dict[str, Any]] = {}
    for modality in ("param_only", "visual_only", "fusion"):
        loader = build_dataloaders(
            modality=modality,
            split="train",
            batch_size=min(int(args.batch_size), 2),
            num_workers=0,
            target_mode=args.target_mode,
            use_cache=True,
            run_id=run_id,
            paths_override=paths,
            validate_integrity=False,
            runtime_build_cfg={
                "img_size": int(args.img_size),
                "clip_len": int(args.clip_len),
                "cache_dtype": str(args.cache_dtype),
            },
        )
        batch = next(iter(loader))
        info: Dict[str, Any] = {"keys": sorted(list(batch.keys()))}
        if "y" in batch and isinstance(batch["y"], torch.Tensor):
            info["y_shape"] = list(batch["y"].shape)
        if "video" in batch and isinstance(batch["video"], torch.Tensor):
            info["video_shape"] = list(batch["video"].shape)
        if "type_id" in batch and isinstance(batch["type_id"], torch.Tensor):
            info["type_id_shape"] = list(batch["type_id"].shape)
        if "severity" in batch and isinstance(batch["severity"], torch.Tensor):
            info["severity_shape"] = list(batch["severity"].shape)
        if "is_ood_masked" in batch and isinstance(batch["is_ood_masked"], torch.Tensor):
            info["is_ood_masked_shape"] = list(batch["is_ood_masked"].shape)
        checks[modality] = info
    return checks


def main() -> None:
    args = parse_args()
    cfg = load_yaml(PROJECT_ROOT / args.config)
    paths = resolve_run_paths(cfg, args.run_id)
    validate_required_inputs(paths)

    holdout_types = parse_holdout_types(args.ood_holdout_types)
    strict_train_allowed_types = parse_train_allowed_types(args.strict_train_allowed_types)
    allow_train_subset = bool(args.allow_train_type_subset_for_smoke and is_smoke_run(args.run_id))
    rebuild_forced = bool(args.rebuild_cache)
    ensure_dir(paths.targets_root)
    ensure_dir(paths.model_inputs_root)
    ensure_dir(paths.evaluation_root)

    cleanup_info = {"removed_files": 0, "removed_dirs": 0}
    if rebuild_forced:
        cleanup_info = cleanup_run_outputs_for_rebuild(
            paths=paths,
            run_id=args.run_id,
            clip_len=int(args.clip_len),
            img_size=int(args.img_size),
            cache_dtype=str(args.cache_dtype),
        )
        print(
            "[INFO] --rebuild_cache enabled; removed stale run outputs: "
            f"files={cleanup_info['removed_files']}, dirs={cleanup_info['removed_dirs']}"
        )

    base_fingerprint = fingerprint_payload(paths=paths, args=args, run_id=args.run_id)
    cache_reused = assert_cache_state_or_raise(
        paths=paths,
        run_id=args.run_id,
        expected_base_fingerprint=base_fingerprint,
        rebuild_cache=rebuild_forced,
    )
    if rebuild_forced:
        cache_reused = False

    target_stats_effective_counts, target_stats_raw_counts, reused_column_missing = (
        load_target_stats_effective_counts(paths.target_stats)
    )

    if cache_reused:
        existing_fingerprint = load_json(get_fingerprint_path(paths, args.run_id))
        artifact_hashes = existing_fingerprint.get("artifacts", {})
        if not isinstance(artifact_hashes, dict):
            artifact_hashes = {}
        report_path = copy_provenance_and_write_report(
            paths=paths,
            run_id=args.run_id,
            fingerprint=existing_fingerprint,
            report_extra={
                "cache_reused": True,
                "message": "Fingerprint matched and artifacts were reused without rebuild.",
                "rebuild_forced": rebuild_forced,
                "cleanup_info": cleanup_info,
                "target_stats_effective_counts": target_stats_effective_counts,
                "target_stats_raw_counts": target_stats_raw_counts,
                "target_stats_required_rows_enforced": is_canonical_run(args.run_id),
                "reused_column_missing": reused_column_missing,
                "artifact_hashes": artifact_hashes,
                "shared_sample_manifest_path": str(get_shared_sample_manifest_path(paths, args.run_id)),
                "shared_sample_count": int(artifact_hashes.get("shared_sample_count", -1))
                if isinstance(artifact_hashes, dict)
                else -1,
                "runtime_integrity_policy": "enabled_by_default",
            },
        )
        print(f"[INFO] Cache fingerprint matched. Reused artifacts for run_id={args.run_id or 'canonical'}")
        print(f"[INFO] Build report: {report_path}")
        return

    print("[INFO] Building master table (fail-fast join + integrity checks)...")
    master = build_master_table(paths)
    master["cache_key"] = master.apply(
        lambda r: stable_sample_key(
            str(r["clip_id"]),
            str(r["degradation_type"]),
            str(r["degradation_param_norm"]),
        ),
        axis=1,
    )

    print("[INFO] Applying proposal-compliant train-only transforms + OOD masking...")
    master, transform_meta = fit_and_apply_transforms(
        master=master,
        ood_policy=args.ood_policy,
        holdout_types=holdout_types,
        strict_train_allowed_types=strict_train_allowed_types,
        allow_train_type_subset=allow_train_subset,
    )

    target_rows = len(pd.read_csv(paths.target_manifest))
    if len(master) != target_rows:
        fail(f"Master row mismatch: len(master)={len(master)} vs len(target_manifest)={target_rows}")
    validate_master_against_target_stats(
        master=master,
        target_stats_effective_counts=target_stats_effective_counts,
        require_all_splits=is_canonical_run(args.run_id),
    )

    print("[INFO] Writing split CSVs and normalization metadata...")
    write_info = write_split_csvs(master=master, paths=paths, run_id=args.run_id, transform_meta=transform_meta)
    quick_purity_assertions(paths=paths, run_id=args.run_id)

    print("[INFO] Building shared video cache...")
    cache_meta = build_shared_video_cache(
        master=master,
        paths=paths,
        run_id=args.run_id,
        clip_len=int(args.clip_len),
        img_size=int(args.img_size),
        cache_dtype=str(args.cache_dtype),
        rebuild_cache=bool(args.rebuild_cache),
    )

    print("[INFO] Building split `.pt` indices for each modality...")
    index_counts = build_split_pt_indices(master=master, paths=paths, run_id=args.run_id)

    shared_manifest_info = build_shared_sample_manifest(
        paths=paths,
        run_id=args.run_id,
        sample_dir=Path(str(cache_meta["sample_dir"])),
        cache_keys=master["cache_key"].astype(str).tolist(),
    )

    final_fingerprint, artifact_hashes = build_final_fingerprint(
        base_fingerprint=base_fingerprint,
        paths=paths,
        run_id=args.run_id,
    )

    fp_path_partial = get_partial_fingerprint_path(paths, args.run_id)
    if fp_path_partial.exists():
        fp_path_partial.unlink()
    with fp_path_partial.open("w", encoding="utf-8") as f:
        json.dump(final_fingerprint, f, indent=2, ensure_ascii=False)

    sanity: Dict[str, Any] = {}
    try:
        if not args.skip_sanity_check:
            print("[INFO] Running one-batch sanity checks...")
            sanity = run_sanity_batches(args=args, run_id=args.run_id, paths=paths)
        fp_path = get_fingerprint_path(paths, args.run_id)
        if fp_path.exists():
            fp_path.unlink()
        shutil.move(str(fp_path_partial), str(fp_path))
    except Exception:
        if fp_path_partial.exists():
            fp_path_partial.unlink()
        raise

    report_path = copy_provenance_and_write_report(
        paths=paths,
        run_id=args.run_id,
        fingerprint=final_fingerprint,
        report_extra={
            "cache_reused": False,
            "rebuild_forced": rebuild_forced,
            "cleanup_info": cleanup_info,
            "master_rows": int(len(master)),
            "write_info": write_info,
            "cache_meta": cache_meta,
            "index_counts": index_counts,
            "masked_rows_total": int((master["is_ood_masked"] == 1).sum()),
            "masked_rows_by_split": {
                s: int(((master["split"] == s) & (master["is_ood_masked"] == 1)).sum())
                for s in SPLITS
            },
            "target_stats_effective_counts": target_stats_effective_counts,
            "target_stats_raw_counts": target_stats_raw_counts,
            "target_stats_required_rows_enforced": is_canonical_run(args.run_id),
            "reused_column_missing": reused_column_missing,
            "strict_train_allowed_types": strict_train_allowed_types,
            "allow_train_type_subset_effective": allow_train_subset,
            "artifact_hashes": artifact_hashes,
            "shared_sample_manifest_path": shared_manifest_info["manifest_path"],
            "shared_sample_count": shared_manifest_info["sample_count"],
            "runtime_integrity_policy": "enabled_by_default",
            "sanity_checks": sanity,
        },
    )

    print(f"[SUCCESS] Master rows: {len(master)}")
    print(f"[SUCCESS] Fingerprint: {fp_path}")
    print(f"[SUCCESS] Build report: {report_path}")


if __name__ == "__main__":
    main()
