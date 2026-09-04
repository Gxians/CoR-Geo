"""Deterministic image transformations with operation-specific seeds."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision

from cor_geo.datasets.panorama_crop import CropSpec, circular_crop
from cor_geo.reproducibility import stable_seed

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def resize_ground_base(
    image: Image.Image,
    height: int,
    panorama_width: int,
) -> Image.Image:
    """Return the deterministic RGB panorama stored by the input cache."""
    return vision.resize(
        image.convert("RGB"),
        [int(height), int(panorama_width)],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )


def resize_satellite_base(image: Image.Image, size: int) -> Image.Image:
    """Return the deterministic RGB satellite image stored by the input cache."""
    return vision.resize(
        image.convert("RGB"),
        [int(size), int(size)],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )


def _rng(*items: object) -> np.random.Generator:
    return np.random.default_rng(stable_seed(*items))


def _probability(seed_items: tuple[object, ...], probability: float) -> bool:
    return bool(_rng(*seed_items).random() < probability)


def _uniform(seed_items: tuple[object, ...], low: float, high: float) -> float:
    return float(_rng(*seed_items).uniform(low, high))


def _color_jitter(
    image: Image.Image,
    key: tuple[object, ...],
    amount: float,
    hue: float,
    probability: float,
) -> Image.Image:
    if not _probability((*key, "color_jitter_apply"), probability):
        return image
    brightness = _uniform((*key, "brightness"), 1.0 - amount, 1.0 + amount)
    contrast = _uniform((*key, "contrast"), 1.0 - amount, 1.0 + amount)
    saturation = _uniform((*key, "saturation"), 1.0 - amount, 1.0 + amount)
    hue_factor = _uniform((*key, "hue"), -hue, hue)
    image = vision.adjust_brightness(image, brightness)
    image = vision.adjust_contrast(image, contrast)
    image = vision.adjust_saturation(image, saturation)
    return vision.adjust_hue(image, hue_factor)


def _gaussian_blur(
    image: Image.Image,
    key: tuple[object, ...],
    sigma_range: tuple[float, float],
    probability: float,
) -> Image.Image:
    if not _probability((*key, "gaussian_blur_apply"), probability):
        return image
    sigma = _uniform((*key, "gaussian_blur_sigma"), *sigma_range)
    return vision.gaussian_blur(image, kernel_size=[5, 5], sigma=[sigma, sigma])


def _normalize(image: Image.Image) -> torch.Tensor:
    return vision.normalize(vision.to_tensor(image), mean=_MEAN, std=_STD)


def _random_erasing(tensor: torch.Tensor, key: tuple[object, ...]) -> torch.Tensor:
    if not _probability((*key, "random_erasing_apply"), 0.1):
        return tensor
    height, width = tensor.shape[-2:]
    generator = _rng(*key, "random_erasing_geometry")
    area = height * width
    for _ in range(10):
        target = area * float(generator.uniform(0.02, 0.08))
        aspect = float(np.exp(generator.uniform(np.log(0.3), np.log(3.3))))
        erase_h = int(round(np.sqrt(target * aspect)))
        erase_w = int(round(np.sqrt(target / aspect)))
        if 0 < erase_h < height and 0 < erase_w < width:
            top = int(generator.integers(0, height - erase_h + 1))
            left = int(generator.integers(0, width - erase_w + 1))
            output = tensor.clone()
            output[:, top : top + erase_h, left : left + erase_w] = 0.0
            return output
    return tensor


@dataclass(frozen=True)
class GroundTransform:
    """Resize, circularly crop, augment, and normalize a panorama."""

    height: int = 378
    panorama_width: int = 756
    dataset_name: str = "cvact"
    target_widths: Mapping[int, int] = field(
        default_factory=lambda: {360: 756, 180: 378, 90: 196, 70: 154}
    )

    def __call__(
        self,
        image: Image.Image,
        crop: CropSpec,
        query_id: str,
        global_seed: int,
        epoch: int,
        fov_deg: int,
        training: bool,
        pre_resized: bool = False,
    ) -> torch.Tensor:
        if pre_resized:
            if image.mode != "RGB" or image.size != (
                self.panorama_width,
                self.height,
            ):
                raise ValueError("Cached ground image has the wrong mode or size")
        else:
            image = resize_ground_base(
                image,
                self.height,
                self.panorama_width,
            )
        image = circular_crop(image, crop.center_px, crop.width)
        target_width = int(self.target_widths[int(fov_deg)])
        if image.width != target_width:
            # Crop at the exact physical FoV first, then make only the small
            # patch-alignment resize required by ViT-B/14 (189->196, 147->154).
            image = vision.resize(
                image,
                [self.height, target_width],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
        key = (global_seed, self.dataset_name, "train", epoch, query_id, fov_deg)
        if training:
            image = _color_jitter(image, key, amount=0.2, hue=0.05, probability=0.8)
            image = _gaussian_blur(image, key, sigma_range=(0.1, 2.0), probability=0.1)
        tensor = _normalize(image)
        return _random_erasing(tensor, key) if training else tensor


@dataclass(frozen=True)
class SatelliteTransform:
    """Resize, deterministically augment, and normalize a satellite image."""

    size: int = 378
    dataset_name: str = "cvact"

    def __call__(
        self,
        image: Image.Image,
        satellite_id: str,
        global_seed: int,
        epoch: int,
        fov_deg: int,
        training: bool,
        pre_resized: bool = False,
    ) -> torch.Tensor:
        if pre_resized:
            if image.mode != "RGB" or image.size != (self.size, self.size):
                raise ValueError("Cached satellite image has the wrong mode or size")
        else:
            image = resize_satellite_base(image, self.size)
        key = (
            global_seed,
            self.dataset_name,
            "train",
            epoch,
            satellite_id,
            fov_deg,
        )
        if training:
            image = _color_jitter(image, key, amount=0.15, hue=0.03, probability=0.5)
            image = _gaussian_blur(image, key, sigma_range=(0.1, 1.5), probability=0.1)
        return _normalize(image)
