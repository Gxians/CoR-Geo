"""Deterministic orientation generation and circular panorama cropping."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from cor_geo.reproducibility import stable_seed

UINT32_RANGE = 2**32


@dataclass(frozen=True)
class CropSpec:
    """A reproducible finite-FoV crop definition."""

    orientation_u32: int
    center_px: int
    width: int
    panorama_width: int

    @property
    def center_deg(self) -> float:
        """Return the continuous protocol orientation in degrees."""
        return 360.0 * self.orientation_u32 / UINT32_RANGE

    @property
    def interval(self) -> tuple[int, int]:
        """Return the unwrapped half-open pixel interval."""
        return self.center_px - self.width // 2, self.center_px + (self.width + 1) // 2


def orientation_to_center_px(orientation_u32: int, panorama_width: int) -> int:
    """Map a uint32 orientation to a panorama pixel using integer arithmetic."""
    if not 0 <= orientation_u32 < UINT32_RANGE:
        raise ValueError(f"orientation_u32 out of range: {orientation_u32}")
    if panorama_width <= 0:
        raise ValueError("panorama_width must be positive")
    return orientation_u32 * panorama_width // UINT32_RANGE


def center_px_to_orientation(center_px: int, panorama_width: int) -> int:
    """Encode an exact panorama pixel center as the smallest matching uint32."""
    if panorama_width <= 0:
        raise ValueError("panorama_width must be positive")
    center = int(center_px) % int(panorama_width)
    return (center * UINT32_RANGE + panorama_width - 1) // panorama_width


def random_crop_spec(
    roll_angle_deg: int,
    fov_deg: int,
    panorama_width: int,
) -> CropSpec:
    """Apply an integer right-roll followed by a left-edge FoV crop."""
    if not 0 <= int(roll_angle_deg) <= 359:
        raise ValueError("Roll angle must be an integer in [0, 359]")
    if not 0 < int(fov_deg) <= 360:
        raise ValueError(f"FoV must be in (0, 360], got {fov_deg}")
    if panorama_width <= 0:
        raise ValueError("panorama_width must be positive")
    if int(fov_deg) * int(panorama_width) % 360:
        raise ValueError(
            f"FoV {fov_deg} does not map to an exact integer width at "
            f"panorama width {panorama_width}"
        )
    roll_pixels = int(roll_angle_deg) * int(panorama_width) // 360
    width = int(fov_deg) * int(panorama_width) // 360
    start_px = (-roll_pixels) % int(panorama_width)
    center_px = (start_px + width // 2) % int(panorama_width)
    orientation = center_px_to_orientation(center_px, int(panorama_width))
    return CropSpec(orientation, center_px, width, int(panorama_width))
def train_crop_spec(
    global_seed: int,
    epoch: int,
    query_id: str,
    fov_deg: int,
    panorama_width: int,
    dataset_name: str = "cvact",
) -> CropSpec:
    """Build a deterministic epoch-specific training crop."""
    orientation = stable_seed(
        global_seed,
        str(dataset_name),
        "train",
        epoch,
        query_id,
        "train_orientation",
    )
    width = panorama_width * fov_deg // 360
    if width * 360 != panorama_width * fov_deg:
        raise ValueError(f"FoV {fov_deg} does not map to an exact integer width")
    return CropSpec(orientation, orientation_to_center_px(orientation, panorama_width), width, panorama_width)


def hard_mining_crop_spec(
    global_seed: int,
    source_epoch: int,
    query_id: str,
    fov_deg: int,
    panorama_width: int,
    dataset_name: str = "cvact",
) -> CropSpec:
    """Return the exact crop shared by pool export and later hard batches."""
    orientation = stable_seed(
        global_seed,
        str(dataset_name),
        "train",
        source_epoch,
        query_id,
        "hard_mining_orientation",
    )
    width = panorama_width * fov_deg // 360
    if width * 360 != panorama_width * fov_deg:
        raise ValueError(f"FoV {fov_deg} does not map to an exact integer width")
    return CropSpec(
        orientation,
        orientation_to_center_px(orientation, panorama_width),
        width,
        panorama_width,
    )


def circular_crop(image: Image.Image, center_px: int, width: int) -> Image.Image:
    """Crop a horizontal interval with circular wrap and no padding."""
    if width <= 0 or width > image.width:
        raise ValueError(f"Crop width must be in [1, {image.width}], got {width}")
    center = center_px % image.width
    start = center - width // 2
    indices = [(start + offset) % image.width for offset in range(width)]
    if indices == list(range(indices[0], indices[0] + width)):
        return image.crop((indices[0], 0, indices[0] + width, image.height))
    left_width = image.width - (start % image.width)
    first = image.crop((start % image.width, 0, image.width, image.height))
    second = image.crop((0, 0, width - left_width, image.height))
    output = Image.new(image.mode, (width, image.height))
    output.paste(first, (0, 0))
    output.paste(second, (first.width, 0))
    return output
