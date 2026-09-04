import torch

from cor_geo.models.content_order_encoder import (
    BilinearSquareRaySampler,
    ConservativeAngularResampler,
    ContentOrderEncoder,
)


def test_conservative_angular_resampler_preserves_constants_and_gradients() -> None:
    resampler = ConservativeAngularResampler(source_bins=54, target_bins=36)
    features = torch.ones(2, 5, 54, 7, requires_grad=True)
    output = resampler(features)
    assert output.shape == (2, 5, 36, 7)
    assert torch.allclose(output, torch.ones_like(output), atol=1.0e-7)
    assert torch.allclose(
        resampler.assignment.sum(dim=1), torch.ones(36), atol=1.0e-7
    )
    output.square().mean().backward()
    assert features.grad is not None


def test_bilinear_square_rays_are_in_bounds_ordered_and_have_expected_shape() -> None:
    sampler = BilinearSquareRaySampler(input_size=27, angular_bins=36, sequence_length=16)
    assert sampler.sampling_grid.shape == (36, 16, 2)
    assert bool((sampler.sampling_grid >= -1.0).all())
    assert bool((sampler.sampling_grid <= 1.0).all())
    # Direction zero is north: x stays centered and y moves monotonically upward.
    assert torch.allclose(sampler.sampling_grid[0, :, 0], torch.zeros(16), atol=1e-6)
    assert bool(torch.all(torch.diff(sampler.sampling_grid[0, :, 1]) < 0))
    assert torch.allclose(sampler.assignment.sum(dim=1), torch.ones(36 * 16))
    nonzero_per_sample = (sampler.assignment > 0.0).sum(dim=1)
    assert int(nonzero_per_sample.max()) <= 4
    assert bool((nonzero_per_sample > 1).any())
    constant = torch.ones(1, 27, 27, 3)
    assert torch.allclose(sampler(constant), torch.ones(1, 36, 16, 3))
    features = torch.randn(2, 27, 27, 32)
    rays = sampler(features)
    assert rays.shape == (2, 36, 16, 32)


def test_content_and_order_subspaces_form_the_joint_descriptor() -> None:
    encoder = ContentOrderEncoder(
        32, 16, content_dim=8, order_dim=8, joint_order_weight=0.2
    )
    columns = torch.randn(2, 7, 16, 32)
    encoded = encoder(columns, return_attention=True)
    attention = encoded.content_attention
    assert attention is not None
    assert encoded.direction.shape == (2, 7, 16)
    assert encoded.content_direction.shape == (2, 7, 8)
    assert encoded.order_direction.shape == (2, 7, 8)
    assert attention.shape == (2, 7, 16)
    assert torch.allclose(attention.sum(dim=-1), torch.ones(2, 7), atol=1e-6)
    projected = encoder.projection(columns)
    content_tokens, order_tokens = projected.split((8, 8), dim=-1)
    expected_content = torch.einsum(
        "bal,bald->bad", attention.to(projected.dtype), content_tokens
    )
    expected_content = torch.nn.functional.normalize(expected_content.float(), dim=-1)
    weights = encoder.first_cosine_weights(16, columns.device)
    expected_order = torch.einsum("l,bald->bad", weights, order_tokens)
    expected_order = torch.nn.functional.normalize(expected_order.float(), dim=-1)
    expected_joint = torch.cat(
        (expected_content * (0.8**0.5), expected_order * (0.2**0.5)), dim=-1
    )
    assert torch.allclose(encoded.content_direction.float(), expected_content, atol=1e-6)
    assert torch.allclose(encoded.order_direction.float(), expected_order, atol=1e-6)
    assert torch.allclose(encoded.direction.float(), expected_joint, atol=1e-6)
    assert torch.allclose(
        encoded.direction.float().norm(dim=-1), torch.ones(2, 7), atol=1e-6
    )


def test_first_cosine_order_reverses_sign_while_content_is_permutation_invariant() -> None:
    encoder = ContentOrderEncoder(32, 16, content_dim=8, order_dim=8)
    columns = torch.randn(2, 5, 16, 32)
    forward = encoder(columns)
    reverse = encoder(columns.flip(dims=(2,)))
    weights = encoder.first_cosine_weights(16, columns.device)
    assert torch.allclose(weights, -weights.flip(0), atol=1e-7)
    assert torch.allclose(weights.sum(), torch.zeros(()), atol=1e-7)
    assert torch.allclose(weights.abs().sum(), torch.ones(()), atol=1e-7)
    assert torch.allclose(
        forward.content_direction, reverse.content_direction, atol=1e-5
    )
    assert torch.allclose(
        forward.order_direction, -reverse.order_direction, atol=1e-5
    )


def test_content_query_and_both_projection_halves_receive_gradients() -> None:
    encoder = ContentOrderEncoder(32, 16, content_dim=8, order_dim=8)
    encoded = encoder(torch.randn(2, 5, 16, 32))
    (encoded.direction[..., 0].sum() + encoded.order_direction[..., 0].sum()).backward()
    assert encoder.pool_query.grad is not None
    projection_gradient = encoder.projection[1].weight.grad
    assert projection_gradient is not None
    assert float(projection_gradient[:8].abs().sum()) > 0.0
    assert float(projection_gradient[8:].abs().sum()) > 0.0
