import pytest
from conftest import TinyDINO

from cor_geo.config import load_experiment_config
from cor_geo.engine.stage_scheduler import GroupCosineScheduler, build_optimizer
from cor_geo.models.cor_geo_model import CoRGeoModel


def test_supported_configuration_is_valid(project_root) -> None:
    config = load_experiment_config(
        project_root / "configs" / "cvact.yaml",
        project_root / "configs" / "model.yaml",
        project_root / "configs" / "train_cvact.yaml",
        project_root / "configs" / "evaluation.yaml",
        project_root / "configs" / "paths.yaml",
    )
    assert config.evaluation["report_split"] == "val"


def test_hard_batches_use_current_cor_geo_model(project_root) -> None:
    config = load_experiment_config(
        project_root / "configs" / "cvact.yaml",
        project_root / "configs" / "model.yaml",
        project_root / "configs" / "train_cvact.yaml",
        project_root / "configs" / "evaluation.yaml",
        project_root / "configs" / "paths.yaml",
    )
    assert config.train["hard_mining"]["pool_source"] == "current_cor_geo_model"
    assert config.train["hard_mining"]["coarse_candidate_locations"] == 512


def test_optimizer_groups_are_complete_and_activate_dino_at_epoch_nine(
    model_config: dict,
    train_config: dict,
) -> None:
    model = CoRGeoModel(model_config, backbone_model=TinyDINO())
    optimizer = build_optimizer(model, train_config)
    ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    names = {group["logical_name"] for group in optimizer.param_groups}
    assert names == {
        "content_order_encoder",
        "dinov2_block_8",
        "dinov2_block_9",
        "dinov2_block_10",
        "dinov2_block_11",
        "dinov2_final_norm",
    }
    scheduler = GroupCosineScheduler(
        optimizer,
        train_config,
        steps_per_epoch=555,
        global_step=8 * 555,
    )
    values = scheduler.set_for_next_step()
    assert values["dinov2_block_11"] == 1.0e-5 / 250
    assert values["dinov2_block_10"] == 5.0e-6 / 250
    assert values["dinov2_block_9"] == 2.5e-6 / 250
    assert values["dinov2_block_8"] == 1.25e-6 / 250


def test_schedule_preserves_prefix_and_uses_two_exact_low_lr_refinements(
    model_config: dict,
    train_config: dict,
) -> None:
    model = CoRGeoModel(model_config, backbone_model=TinyDINO())
    optimizer = build_optimizer(model, train_config)
    scheduler = GroupCosineScheduler(optimizer, train_config, steps_per_epoch=555)

    scheduler.global_step = 48 * 555 - 1
    values = scheduler.set_for_next_step()
    assert values["content_order_encoder"] == pytest.approx(2.0e-6)
    assert values["dinov2_block_11"] == pytest.approx(1.0e-6)

    scheduler.global_step = 48 * 555
    values = scheduler.set_for_next_step()
    assert values["content_order_encoder"] == pytest.approx(2.0e-5)
    assert values["dinov2_block_8"] == pytest.approx(2.5e-7)

    scheduler.global_step = 56 * 555 - 1
    values = scheduler.set_for_next_step()
    assert values["content_order_encoder"] == pytest.approx(2.0e-6)
    assert values["dinov2_block_11"] == pytest.approx(1.0e-6)

    scheduler.global_step = 56 * 555
    values = scheduler.set_for_next_step()
    assert values["content_order_encoder"] == pytest.approx(1.0e-5)
    assert values["dinov2_block_8"] == pytest.approx(1.25e-7)

    scheduler.global_step = 64 * 555 - 1
    values = scheduler.set_for_next_step()
    assert values["content_order_encoder"] == pytest.approx(1.0e-6)
    assert values["dinov2_block_11"] == pytest.approx(5.0e-7)
