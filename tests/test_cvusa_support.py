import torch
from conftest import TinyDINO

from cor_geo.config import load_experiment_config, load_yaml
from cor_geo.datasets.panorama_crop import train_crop_spec
from cor_geo.models.cor_geo_model import CoRGeoModel


def test_cvusa_and_cvact_share_input_and_angular_geometry(project_root) -> None:
    eval_config = project_root / "configs" / "evaluation.yaml"
    paths_config = project_root / "configs" / "paths.yaml"
    cvact = load_experiment_config(
        project_root / "configs" / "cvact.yaml",
        project_root / "configs" / "model.yaml",
        project_root / "configs" / "train_cvact.yaml",
        eval_config,
        paths_config,
    )
    cvusa = load_experiment_config(
        project_root / "configs" / "cvusa.yaml",
        project_root / "configs" / "model.yaml",
        project_root / "configs" / "train_cvusa.yaml",
        eval_config,
        paths_config,
    )
    assert cvact.model["input"]["ground_height"] == 224
    assert cvusa.model["input"]["ground_height"] == 224
    assert cvact.model["input"]["panorama_width"] == 756
    assert cvusa.model["input"]["panorama_width"] == 756
    assert cvact.model["input"]["widths"] == cvusa.model["input"]["widths"]
    assert cvact.model["input"]["satellite_size"] == [378, 378]
    assert cvusa.model["input"]["satellite_size"] == [378, 378]
    cvact_column = cvact.model["architecture"]["content_order_encoder"]
    cvusa_column = cvusa.model["architecture"]["content_order_encoder"]
    assert cvact_column["ground_sequence_length"] == 16
    assert cvusa_column["ground_sequence_length"] == 16
    assert cvact_column["satellite_sequence_length"] == 16
    assert cvusa_column["satellite_sequence_length"] == 16
    assert cvact.train["stages"] == cvusa.train["stages"]
    assert cvact.train["dataset_cache"]["root"] != cvusa.train["dataset_cache"]["root"]
    assert cvusa.dataset["manifest_splits"] == ["train", "val"]


def test_training_crop_is_namespaced_by_dataset() -> None:
    cvact = train_crop_spec(7, 1, "0001227", 70, 756, "cvact")
    cvusa = train_crop_spec(7, 1, "0001227", 70, 756, "cvusa")
    assert cvact.width == cvusa.width == 147
    assert cvact.orientation_u32 != cvusa.orientation_u32


def test_cvusa_shares_one_encoder_across_ground_and_satellite_16_tokens(
    project_root,
) -> None:
    model_config = load_yaml(project_root / "configs" / "model.yaml")
    model = CoRGeoModel(model_config, backbone_model=TinyDINO()).eval()
    with torch.no_grad():
        ground = model.encode_ground(torch.randn(1, 3, 224, 756), 360)
        satellite = model.encode_satellite(torch.randn(1, 3, 378, 378))
    assert model.ground_sequence_length == 16
    assert model.satellite_sequence_length == 16
    assert model.satellite_ray_sampler.sequence_length == 16
    assert ground.direction.shape == satellite.direction.shape == (1, 36, 256)
