import torch

from cor_geo.losses import info_nce


def test_metadata_all_gather_materializes_expanded_views(monkeypatch) -> None:
    expanded = torch.tensor(0.6).expand(8)
    assert not expanded.is_contiguous()

    monkeypatch.setattr(info_nce.distributed, "is_available", lambda: True)
    monkeypatch.setattr(info_nce.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(info_nce.distributed, "get_world_size", lambda: 2)

    def fake_all_gather(outputs, tensor) -> None:
        assert tensor.is_contiguous()
        outputs[0].copy_(tensor)
        outputs[1].copy_(tensor)

    monkeypatch.setattr(info_nce.distributed, "all_gather", fake_all_gather)
    gathered = info_nce.metadata_all_gather(expanded)

    assert gathered.shape == (16,)
    assert gathered.is_contiguous()
    assert torch.allclose(gathered, torch.full((16,), 0.6))
