"""CVACT/CVUSA annotation parsing, immutable manifests, and strict audits."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat

from cor_geo.config import load_yaml, resolve_project_path
from cor_geo.utils.hashing import sha256_file, sha256_json
from cor_geo.utils.io import write_json

MANIFEST_COLUMNS = [
    "dataset",
    "split",
    "query_id",
    "query_path",
    "satellite_id",
    "satellite_path",
    "source_mat_struct",
    "source_mat_index",
]


def project_relative_path(path: str | Path, project_root: str | Path) -> str:
    """Return a portable POSIX path and reject files outside the repository."""
    resolved_root = Path(project_root).expanduser().resolve()
    resolved_path = Path(path).expanduser().resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(
            f"Dataset path must be inside the repository: {resolved_path}"
        ) from error
    return relative.as_posix()


def _resolve_manifest_image_path(value: object, project_root: Path) -> str:
    relative = Path(str(value))
    if relative.is_absolute():
        raise ValueError("Manifest image paths must be repository-relative")
    resolved = resolve_project_path(relative, project_root)
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as error:
        raise ValueError(
            f"Manifest image path escapes the repository: {relative}"
        ) from error
    return str(resolved)


def _require_parquet_engine() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError as error:
        raise RuntimeError("pyarrow 17.0.0 is required for immutable Parquet manifests") from error


def write_parquet_atomic(frame: pd.DataFrame, path: str | Path, overwrite: bool = False) -> Path:
    """Atomically write a Parquet dataframe."""
    _require_parquet_engine()
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing manifest: {resolved}")
    descriptor, name = tempfile.mkstemp(prefix=f".{resolved.name}.", suffix=".parquet", dir=resolved.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        frame.to_parquet(temporary, index=False, engine="pyarrow")
        temporary.replace(resolved)
    finally:
        temporary.unlink(missing_ok=True)
    return resolved


def read_manifest(path: str | Path) -> pd.DataFrame:
    """Read a manifest and resolve its repository-relative image paths."""
    _require_parquet_engine()
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    frame = pd.read_parquet(resolved, engine="pyarrow")
    missing = [column for column in MANIFEST_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Manifest is missing columns {missing}: {resolved}")
    manifest_roots = [parent for parent in resolved.parents if parent.name == "data_manifests"]
    if len(manifest_roots) != 1:
        raise ValueError(f"Manifest must live under data_manifests/: {resolved}")
    project_root = manifest_roots[0].parent
    for column in ("query_path", "satellite_path"):
        frame[column] = frame[column].map(
            lambda value: _resolve_manifest_image_path(value, project_root)
        )
    return frame


def parquet_row_count(path: str | Path) -> int:
    """Read a Parquet row count from file metadata without assuming a data-column name."""
    _require_parquet_engine()
    import pyarrow.parquet as parquet

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    metadata = parquet.ParquetFile(resolved).metadata
    if metadata is None:
        raise ValueError(f"Parquet metadata is unavailable: {resolved}")
    return int(metadata.num_rows)


def _load_annotations(dataset_root: Path, dataset_config: dict[str, Any]) -> dict[str, list[tuple[int, str]]]:
    mat_path = dataset_root / dataset_config["mat_file"]
    if not mat_path.is_file():
        raise FileNotFoundError(mat_path)
    values = loadmat(mat_path, simplify_cells=True)
    pano_ids = np.asarray(values["panoIds"]).reshape(-1)
    output: dict[str, list[tuple[int, str]]] = {}
    for split, split_config in dataset_config["splits"].items():
        struct = split_config["mat_struct"]
        field = split_config["mat_index_field"]
        indices = np.asarray(values[struct][field]).reshape(-1)
        if len(indices) != split_config["expected_annotated_count"]:
            raise ValueError(f"{split} annotation count is {len(indices)}, expected {split_config['expected_annotated_count']}")
        rows: list[tuple[int, str]] = []
        for raw_index in indices:
            matlab_index = int(raw_index)
            if not 1 <= matlab_index <= len(pano_ids):
                raise ValueError(f"MATLAB index out of bounds in {split}: {matlab_index}")
            rows.append((matlab_index, str(pano_ids[matlab_index - 1])))
        output[split] = rows
    return output


def _load_exclusions(project_root: Path, dataset_config: dict[str, Any]) -> list[dict[str, Any]]:
    path = resolve_project_path(dataset_config["exclusions_config"], project_root)
    config = load_yaml(path)
    exclusions = config.get("exclusions")
    if not isinstance(exclusions, list):
        raise ValueError("exclusions must be a list")
    return exclusions


def _paths_for_id(
    dataset_root: Path,
    split_config: dict[str, Any],
    patterns: dict[str, str],
    query_id: str,
) -> tuple[Path, Path]:
    substitutions = {
        "image_root": split_config["image_root"],
        "query_id": query_id,
        "satellite_id": query_id,
    }
    return (
        dataset_root / patterns["query"].format(**substitutions),
        dataset_root / patterns["satellite"].format(**substitutions),
    )


def _verify_image(path: Path) -> tuple[str, tuple[int, int]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with Image.open(path) as image:
            mode = image.mode
            size = image.size
            if mode not in {"1", "L", "LA", "P", "RGB", "RGBA", "CMYK", "YCbCr", "I", "F"}:
                raise ValueError(f"Unsupported PIL mode {mode}")
            image.verify()
            return path.suffix.lower(), size
    except Exception as error:
        raise ValueError(f"Unreadable or non-RGB-compatible image: {path}") from error


def _ids_from_directory(directory: Path, suffix: str) -> set[str]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    ids: set[str] = set()
    for path in directory.iterdir():
        if path.is_file() and path.name.endswith(suffix):
            ids.add(path.name[: -len(suffix)])
    return ids


def _validate_split_sets(frames: dict[str, pd.DataFrame]) -> None:
    if not {"train", "val"}.issubset(frames):
        raise ValueError("Pair manifests must contain train and val splits")
    sets = {split: set(frame["query_id"].astype(str)) for split, frame in frames.items()}
    if sets["train"] & sets["val"]:
        raise ValueError("train and val IDs must be disjoint")
    if "test" in sets:
        if sets["train"] & sets["test"]:
            raise ValueError("train IDs must be disjoint from test")
        if not sets["val"] < sets["test"]:
            raise ValueError("CVACT val must be a strict subset of test")


def build_cvact_manifests(
    dataset_config_path: str | Path,
    paths_config_path: str | Path,
    overwrite: bool = False,
    verify_images: bool = True,
    progress_every: int = 10_000,
) -> dict[str, Path]:
    """Create all pair manifests after exhaustive CVACT integrity checks."""
    dataset_config = load_yaml(dataset_config_path)
    paths_config = load_yaml(paths_config_path)
    project_root = Path(paths_config["project_root"]).expanduser().resolve()
    dataset_root = resolve_project_path(
        paths_config["datasets"][dataset_config["root_key"]],
        project_root,
    )
    output_root = project_root / "data_manifests" / "cvact"
    annotations = _load_annotations(dataset_root, dataset_config)
    exclusions = _load_exclusions(project_root, dataset_config)
    exclusion_keys = {(row["split"], str(row["query_id"])) for row in exclusions}
    if len(exclusion_keys) != len(exclusions):
        raise ValueError("Duplicate exclusion entries are forbidden")
    applied: list[dict[str, Any]] = []
    frames: dict[str, pd.DataFrame] = {}
    dimensions: Counter[str] = Counter()
    extensions: Counter[str] = Counter()
    verified_images = 0
    for split, annotated_rows in annotations.items():
        split_config = dataset_config["splits"][split]
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for matlab_index, query_id in annotated_rows:
            if query_id in seen:
                raise ValueError(f"Duplicate query_id in {split}: {query_id}")
            seen.add(query_id)
            query_path, satellite_path = _paths_for_id(
                dataset_root,
                split_config,
                dataset_config["path_patterns"],
                query_id,
            )
            if (split, query_id) in exclusion_keys:
                exclusion = next(row for row in exclusions if row["split"] == split and row["query_id"] == query_id)
                if int(exclusion["source_mat_index"]) != matlab_index:
                    raise ValueError(f"Exclusion source index mismatch for {query_id}")
                applied.append(
                    {
                        **exclusion,
                        "query_exists": query_path.is_file(),
                        "satellite_exists": satellite_path.is_file(),
                    }
                )
                continue
            if not query_path.is_file() or not satellite_path.is_file():
                raise FileNotFoundError(f"Unapproved missing pair in {split}: {query_id}")
            if verify_images:
                for image_path in (query_path, satellite_path):
                    extension, size = _verify_image(image_path)
                    extensions[extension] += 1
                    dimensions[f"{size[0]}x{size[1]}"] += 1
                    verified_images += 1
                    if progress_every > 0 and verified_images % progress_every == 0:
                        print(f"verified_images={verified_images}", flush=True)
            records.append(
                {
                    "dataset": "cvact",
                    "split": split,
                    "query_id": query_id,
                    "query_path": project_relative_path(query_path, project_root),
                    "satellite_id": query_id,
                    "satellite_path": project_relative_path(
                        satellite_path,
                        project_root,
                    ),
                    "source_mat_struct": split_config["mat_struct"],
                    "source_mat_index": matlab_index,
                }
            )
        frame = pd.DataFrame(records, columns=MANIFEST_COLUMNS).sort_values("query_id").reset_index(drop=True)
        if len(frame) != split_config["expected_final_count"]:
            raise ValueError(f"{split} final count is {len(frame)}, expected {split_config['expected_final_count']}")
        if not frame["query_id"].is_unique or not frame["satellite_id"].is_unique:
            raise ValueError(f"IDs must be unique within {split}")
        if not (frame["query_id"] == frame["satellite_id"]).all():
            raise ValueError(f"query_id/satellite_id mapping is not one-to-one in {split}")
        frames[split] = frame
    if set(exclusion_keys) != {(row["split"], row["query_id"]) for row in applied}:
        raise ValueError("Every configured exclusion must match an annotation")
    _validate_split_sets(frames)
    test_ids = set(frames["test"]["query_id"])
    test_root = dataset_root / dataset_config["splits"]["test"]["image_root"]
    query_files = _ids_from_directory(test_root / "streetview", "_grdView.jpg")
    satellite_files = _ids_from_directory(test_root / "satview_polish", "_satView_polish.jpg")
    if query_files != test_ids or satellite_files != test_ids:
        raise ValueError(
            "ANU_data_test file IDs must exactly equal valSetAll; "
            f"query extra/missing={len(query_files - test_ids)}/{len(test_ids - query_files)}, "
            f"satellite extra/missing={len(satellite_files - test_ids)}/{len(test_ids - satellite_files)}"
        )
    output_paths = {split: output_root / f"{split}.parquet" for split in frames}
    for split, frame in frames.items():
        write_parquet_atomic(frame, output_paths[split], overwrite=overwrite)
    exclusions_path = output_root / "exclusions_applied.parquet"
    write_parquet_atomic(pd.DataFrame(applied), exclusions_path, overwrite=overwrite)
    inventory = {
        "dataset_root": project_relative_path(dataset_root, project_root),
        "path_storage": "repository_relative_posix",
        "annotated_counts": {split: len(rows) for split, rows in annotations.items()},
        "final_counts": {split: len(frame) for split, frame in frames.items()},
        "image_dimensions": dict(sorted(dimensions.items())),
        "extensions": dict(sorted(extensions.items())),
        "test_query_file_count": len(query_files),
        "test_satellite_file_count": len(satellite_files),
    }
    inventory_path = output_root / "inventory.json"
    write_json(inventory_path, inventory, overwrite=overwrite)
    metadata = {
        "protocol_version": "cvact_v2.1",
        "dataset_config_sha256": sha256_file(dataset_config_path),
        "paths_config_sha256": sha256_file(paths_config_path),
        "manifest_sha256": {split: sha256_file(path) for split, path in output_paths.items()},
        "exclusions_sha256": sha256_file(exclusions_path),
        "inventory_sha256": sha256_json(inventory),
    }
    metadata_path = output_root / "manifest_metadata.json"
    write_json(metadata_path, metadata, overwrite=overwrite)
    return {**output_paths, "exclusions": exclusions_path, "inventory": inventory_path, "metadata": metadata_path}


def _resolve_csv_path(dataset_root: Path, raw_path: str) -> Path:
    """Resolve one dataset-relative CSV path without allowing root escape."""
    relative = Path(str(raw_path))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"CVUSA CSV path must be dataset-relative: {raw_path}")
    resolved = (dataset_root / relative).resolve()
    try:
        resolved.relative_to(dataset_root)
    except ValueError as error:
        raise ValueError(f"CVUSA CSV path escapes the dataset root: {raw_path}") from error
    return resolved


def _cvusa_id(path_value: str) -> str:
    stem = Path(str(path_value)).stem
    if not stem.isdigit() or len(stem) != 7:
        raise ValueError(f"CVUSA IDs must be seven decimal digits: {path_value}")
    return stem


def build_cvusa_manifests(
    dataset_config_path: str | Path,
    paths_config_path: str | Path,
    overwrite: bool = False,
    verify_images: bool = True,
    progress_every: int = 10_000,
) -> dict[str, Path]:
    """Create CVUSA train/val pair manifests from the official 19zl split CSVs.

    Source CSV row order is deliberately retained for stable gallery indexing.
    """
    dataset_config = load_yaml(dataset_config_path)
    if dataset_config.get("dataset") != "cvusa":
        raise ValueError("build_cvusa_manifests requires a CVUSA dataset config")
    paths_config = load_yaml(paths_config_path)
    project_root = Path(paths_config["project_root"]).expanduser().resolve()
    dataset_root = resolve_project_path(
        paths_config["datasets"][dataset_config["root_key"]],
        project_root,
    )
    output_root = project_root / "data_manifests" / "cvusa"
    expected_sizes = {
        key: tuple(map(int, value))
        for key, value in dataset_config["expected_source_image_sizes"].items()
    }
    frames: dict[str, pd.DataFrame] = {}
    dimensions: Counter[str] = Counter()
    extensions: Counter[str] = Counter()
    csv_hashes: dict[str, str] = {}
    verified_images = 0
    for split in ("train", "val"):
        split_config = dataset_config["splits"][split]
        csv_path = dataset_root / str(split_config["csv_file"])
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        csv_hashes[split] = sha256_file(csv_path)
        source = pd.read_csv(csv_path, header=None, dtype=str, keep_default_na=False)
        if source.shape[1] != 3:
            raise ValueError(f"{split} CSV must contain satellite, ground, annotation columns")
        if len(source) != int(split_config["expected_count"]):
            raise ValueError(
                f"{split} CSV count is {len(source)}, expected {split_config['expected_count']}"
            )
        records: list[dict[str, Any]] = []
        for row_index, values in enumerate(source.itertuples(index=False, name=None), start=1):
            satellite_raw, query_raw, annotation_raw = map(str, values)
            satellite_id = _cvusa_id(satellite_raw)
            query_id = _cvusa_id(query_raw)
            annotation_id = _cvusa_id(annotation_raw)
            if satellite_id != query_id or query_id != annotation_id:
                raise ValueError(
                    f"CVUSA row {split}:{row_index} does not form an ID-matched triplet"
                )
            satellite_path = _resolve_csv_path(dataset_root, satellite_raw)
            query_path = _resolve_csv_path(dataset_root, query_raw)
            annotation_path = _resolve_csv_path(dataset_root, annotation_raw)
            for role, image_path in (
                ("query", query_path),
                ("satellite", satellite_path),
                ("annotation", annotation_path),
            ):
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                if verify_images:
                    extension, size = _verify_image(image_path)
                    if size != expected_sizes[role]:
                        raise ValueError(
                            f"Unexpected CVUSA {role} size at {image_path}: "
                            f"{size}, expected {expected_sizes[role]}"
                        )
                    extensions[f"{role}:{extension}"] += 1
                    dimensions[f"{role}:{size[0]}x{size[1]}"] += 1
                    verified_images += 1
                    if progress_every > 0 and verified_images % progress_every == 0:
                        print(f"verified_images={verified_images}", flush=True)
            records.append(
                {
                    "dataset": "cvusa",
                    "split": split,
                    "query_id": query_id,
                    "query_path": project_relative_path(query_path, project_root),
                    "satellite_id": satellite_id,
                    "satellite_path": project_relative_path(
                        satellite_path,
                        project_root,
                    ),
                    # Retain the established manifest schema while recording
                    # the CSV source rather than pretending it came from MAT.
                    "source_mat_struct": str(split_config["csv_file"]),
                    "source_mat_index": row_index,
                }
            )
        frame = pd.DataFrame(records, columns=MANIFEST_COLUMNS).reset_index(drop=True)
        if frame["query_id"].duplicated().any() or frame["satellite_id"].duplicated().any():
            raise ValueError(f"Duplicate CVUSA IDs in {split}")
        if not (frame["query_id"] == frame["satellite_id"]).all():
            raise ValueError(f"Broken CVUSA ground/satellite bijection in {split}")
        frames[split] = frame
    _validate_split_sets(frames)

    referenced_ids = set(pd.concat([frames["train"], frames["val"]])["satellite_id"])
    satellite_directory = dataset_root / str(dataset_config["satellite_inventory_root"])
    disk_ids = {
        path.stem
        for path in satellite_directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".jpg"
    }
    if dataset_config.get("strict_satellite_file_equality", False) and disk_ids != referenced_ids:
        raise ValueError(
            "CVUSA CSV union must exactly equal the bingmap/19 inventory; "
            f"extra/missing={len(disk_ids - referenced_ids)}/{len(referenced_ids - disk_ids)}"
        )

    output_paths = {split: output_root / f"{split}.parquet" for split in frames}
    for split, frame in frames.items():
        write_parquet_atomic(frame, output_paths[split], overwrite=overwrite)
    inventory = {
        "dataset_root": project_relative_path(dataset_root, project_root),
        "path_storage": "repository_relative_posix",
        "source_csv_counts": {split: len(frame) for split, frame in frames.items()},
        "source_csv_sha256": csv_hashes,
        "source_image_dimensions": dict(sorted(dimensions.items())),
        "source_extensions": dict(sorted(extensions.items())),
        "referenced_satellite_count": len(referenced_ids),
        "satellite_inventory_count": len(disk_ids),
        "manifest_order": "source_csv_row_order",
        "model_input_geometry": "shared with CVACT in configs/model.yaml",
    }
    inventory_path = output_root / "inventory.json"
    write_json(inventory_path, inventory, overwrite=overwrite)
    metadata = {
        "protocol_version": str(dataset_config["protocol_version"]),
        "dataset_config_sha256": sha256_file(dataset_config_path),
        "paths_config_sha256": sha256_file(paths_config_path),
        "source_csv_sha256": csv_hashes,
        "manifest_sha256": {
            split: sha256_file(path) for split, path in output_paths.items()
        },
        "inventory_sha256": sha256_json(inventory),
    }
    metadata_path = output_root / "manifest_metadata.json"
    write_json(metadata_path, metadata, overwrite=overwrite)
    return {**output_paths, "inventory": inventory_path, "metadata": metadata_path}


def validate_manifest_frames(frames: dict[str, pd.DataFrame], expected_counts: dict[str, int]) -> None:
    """Validate loaded frames without changing them."""
    for split, frame in frames.items():
        if len(frame) != expected_counts[split]:
            raise ValueError(f"Unexpected {split} manifest count: {len(frame)}")
        if frame["query_id"].duplicated().any() or frame["satellite_id"].duplicated().any():
            raise ValueError(f"Duplicate IDs in {split} manifest")
        if not (frame["query_id"].astype(str) == frame["satellite_id"].astype(str)).all():
            raise ValueError(f"Broken query/satellite bijection in {split}")
        missing_paths = [path for path in _paths(frame) if not Path(path).is_file()]
        if missing_paths:
            raise FileNotFoundError(f"{split} manifest has missing paths, first={missing_paths[0]}")
    _validate_split_sets(frames)


def _paths(frame: pd.DataFrame) -> Iterable[str]:
    yield from frame["query_path"].astype(str)
    yield from frame["satellite_path"].astype(str)


def load_manifest_metadata(path: str | Path) -> dict[str, Any]:
    """Load generated JSON metadata."""
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with resolved.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid manifest metadata: {resolved}")
    return value
