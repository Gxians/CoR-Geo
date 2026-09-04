import torch
from conftest import TinyDINO

from cor_geo.engine.fov_generalization import (
    register_evaluation_resamplers,
    resolve_evaluation_fov_geometry,
)
from cor_geo.losses.info_nce import CoRGeoLoss
from cor_geo.models.cor_geo_model import CoRGeoModel


def _inputs(model_config: dict) -> tuple[dict, dict, torch.Tensor]:
    fovs = (360, 180, 90, 70)
    widths = {int(key): int(value) for key, value in model_config["input"]["widths"].items()}
    ground = {
        fov: torch.randn(1, 3, model_config["input"]["ground_height"], widths[fov])
        for fov in fovs
    }
    positions = {fov: torch.tensor([index]) for index, fov in enumerate(fovs)}
    satellite_size = int(model_config["input"]["satellite_size"][0])
    satellite = torch.randn(4, 3, satellite_size, satellite_size)
    return ground, positions, satellite


def test_mixed_fov_direction_shapes(model_config: dict) -> None:
    model = CoRGeoModel(model_config, backbone_model=TinyDINO())
    model.eval()
    with torch.no_grad():
        output = model(*_inputs(model_config))
    assert output.ground.direction.shape == (4, 36, 256)
    assert output.ground.order_direction.shape == (4, 36, 128)
    assert output.ground.valid.sum(dim=1).tolist() == [36, 18, 9, 7]
    assert output.satellite.direction.shape == (4, 36, 256)
    assert output.satellite.order_direction.shape == (4, 36, 128)
    assert model.content_order_encoder.pool_query.shape == (128,)


def test_parameter_free_unseen_fov_geometry(model_config: dict) -> None:
    model = CoRGeoModel(model_config, backbone_model=TinyDINO())
    geometries = resolve_evaluation_fov_geometry((270, 120), model_config)
    assert geometries[270].physical_crop_width == 567
    assert geometries[270].aligned_input_width == 574
    assert geometries[270].source_patch_columns == 41
    assert geometries[270].target_direction_bins == 27
    assert geometries[120].physical_crop_width == 252
    assert geometries[120].aligned_input_width == 252
    assert geometries[120].source_patch_columns == 18
    assert geometries[120].target_direction_bins == 12
    register_evaluation_resamplers(model, geometries, torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        representation_270 = model.encode_ground(torch.randn(1, 3, 224, 574), 270)
        representation_120 = model.encode_ground(torch.randn(1, 3, 224, 252), 120)
    assert representation_270.valid.sum().item() == 27
    assert representation_120.valid.sum().item() == 12


def test_epoch_one_updates_content_order_encoder_but_not_dino(model_config: dict) -> None:
    model = CoRGeoModel(model_config, backbone_model=TinyDINO())
    model.train()
    model.set_train_epoch(1)
    output = model(*_inputs(model_config))
    CoRGeoLoss()(output).total.backward()
    assert model.content_order_encoder.pool_query.grad is not None
    assert all(
        parameter.grad is None for block in model.backbone.blocks for parameter in block.parameters()
    )


def test_epoch_nine_updates_last_four_dino_blocks(model_config: dict) -> None:
    model = CoRGeoModel(model_config, backbone_model=TinyDINO())
    model.train()
    model.set_train_epoch(9)
    output = model(*_inputs(model_config))
    CoRGeoLoss()(output).total.backward()
    assert all(
        parameter.grad is None for block in model.backbone.blocks[:8] for parameter in block.parameters()
    )
    assert all(
        parameter.grad is not None for block in model.backbone.blocks[8:] for parameter in block.parameters()
    )


def test_total_loss_is_joint_plus_order(model_config: dict) -> None:
    model = CoRGeoModel(model_config, backbone_model=TinyDINO())
    model.eval()
    losses = CoRGeoLoss(order_retrieval_weight=0.15)(
        model(*_inputs(model_config))
    )
    assert torch.allclose(
        losses.total.detach(),
        losses.joint_retrieval
        + 0.15 * losses.order_retrieval
    )
