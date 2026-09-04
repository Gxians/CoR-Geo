from pathlib import Path

import pandas as pd
import pytest

from cor_geo.config import load_yaml
from cor_geo.datasets.manifests import (
    MANIFEST_COLUMNS,
    project_relative_path,
    read_manifest,
    write_parquet_atomic,
)


def test_default_paths_stay_inside_repository(project_root: Path) -> None:
    paths = load_yaml(project_root / "configs" / "paths.yaml")
    assert paths["project_root"] == "."
    assert paths["data_root"] == "data"
    assert paths["output_root"] == "outputs"
    assert paths["datasets"] == {
        "cvact": "data/CVACT",
        "cvusa": "data/CVUSA",
    }
    assert all(
        not Path(value).is_absolute()
        for value in (
            paths["data_root"],
            paths["output_root"],
            paths["dinov2_root"],
            *paths["datasets"].values(),
            *paths["checkpoints"].values(),
        )
    )


def test_manifest_paths_are_repository_relative(project_root: Path) -> None:
    image = project_root / "data" / "CVACT" / "example.jpg"
    assert project_relative_path(image, project_root) == "data/CVACT/example.jpg"


def test_manifest_paths_reject_external_files(
    project_root: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="inside the repository"):
        project_relative_path(tmp_path / "outside.jpg", project_root)


def _manifest_row(query_path: str, satellite_path: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "cvact",
                "split": "val",
                "query_id": "query",
                "query_path": query_path,
                "satellite_id": "query",
                "satellite_path": satellite_path,
                "source_mat_struct": "valSet",
                "source_mat_index": 1,
            }
        ],
        columns=MANIFEST_COLUMNS,
    )


def test_read_manifest_resolves_only_repository_relative_images(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "data_manifests" / "cvact" / "val.parquet"
    write_parquet_atomic(
        _manifest_row("data/CVACT/query.jpg", "data/CVACT/satellite.jpg"),
        manifest_path,
    )
    loaded = read_manifest(manifest_path)
    assert loaded.loc[0, "query_path"] == str(tmp_path / "data" / "CVACT" / "query.jpg")


@pytest.mark.parametrize("unsafe", ["/outside/query.jpg", "../../outside/query.jpg"])
def test_read_manifest_rejects_nonportable_image_paths(
    tmp_path: Path,
    unsafe: str,
) -> None:
    manifest_path = tmp_path / "data_manifests" / "cvact" / "val.parquet"
    write_parquet_atomic(
        _manifest_row(unsafe, "data/CVACT/satellite.jpg"),
        manifest_path,
    )
    with pytest.raises(ValueError, match="repository-relative|escapes the repository"):
        read_manifest(manifest_path)
