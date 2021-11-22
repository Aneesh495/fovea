"""
Sensor calibration stages: black level subtraction, optical lens shading, and defective pixel correction.
"""
import math
from typing import List, Tuple
from ..io.raw import RawBayerImage

def apply_black_level_subtraction(
    image: RawBayerImage,
    black_level: float = 64.0,
    white_level: float = 4095.0
) -> List[float]:
    """Linearizes raw sensor measurements to normalized floating-point range [0.0, 1.0]."""
    scale = white_level - black_level
    if scale <= 0.0:
        raise ValueError("White level must be strictly greater than black level")

    inv_scale = 1.0 / scale
    out = [0.0] * len(image.data)
    for i, val in enumerate(image.data):
        linearized = (val - black_level) * inv_scale
        out[i] = min(1.0, max(0.0, linearized))
    return out

def apply_lens_shading_correction(
    linear_data: List[float],
    width: int,
    height: int,
    radial_poly: List[float] = [1.0, 0.0, 0.0, 0.0],
    center: Tuple[float, float] = None
) -> List[float]:
    """Corrects cosine-fourth and pupil vignetting using a radial polynomial gain map.

    gain(r) = k0 + k1 * r^2 + k2 * r^4 + k3 * r^6, where r is normalized distance from optical center.
    """
    if center is None:
        cx = width / 2.0
        cy = height / 2.0
    else:
        cx, cy = center

    max_dist = math.hypot(max(cx, width - cx), max(cy, height - cy))
    inv_max_dist = 1.0 / max_dist if max_dist > 0 else 1.0

    k0 = radial_poly[0] if len(radial_poly) > 0 else 1.0
    k1 = radial_poly[1] if len(radial_poly) > 1 else 0.0
    k2 = radial_poly[2] if len(radial_poly) > 2 else 0.0
    k3 = radial_poly[3] if len(radial_poly) > 3 else 0.0

    out = [0.0] * len(linear_data)
    for y in range(height):
        dy = y - cy
        for x in range(width):
            dx = x - cx
            r_norm = math.hypot(dx, dy) * inv_max_dist
            r2 = r_norm * r_norm
            r4 = r2 * r2
            r6 = r4 * r2
            gain = k0 + k1 * r2 + k2 * r4 + k3 * r6
            idx = y * width + x
            out[idx] = min(1.0, max(0.0, linear_data[idx] * gain))
    return out

def apply_defective_pixel_correction(
    raw_data: List[float],
    width: int,
    height: int,
    threshold: float = 0.25
) -> List[float]:
    """Detects and replaces stuck hot and dead pixels using same-channel 3x3 neighborhood median."""
    out = list(raw_data)
    # A Bayer pattern has same-color pixels separated by stride 2 in x and y
    for y in range(2, height - 2):
        for x in range(2, width - 2):
            idx = y * width + x
            val = raw_data[idx]

            # 4 cross neighbors of the same color channel
            n_top = raw_data[(y - 2) * width + x]
            n_bot = raw_data[(y + 2) * width + x]
            n_left = raw_data[y * width + (x - 2)]
            n_right = raw_data[y * width + (x + 2)]

            neighbors = sorted([n_top, n_bot, n_left, n_right])
            median = 0.5 * (neighbors[1] + neighbors[2])
            diff = abs(val - median)

            if diff > threshold:
                out[idx] = median
    return out
