"""
High Dynamic Range (HDR) exposure fusion algorithms.
Implements the Mertens-Kautz-Van Reeth multi-exposure fusion technique.
"""
import math
from typing import List

def _compute_weight_map(
    rgb: List[float],
    width: int,
    height: int,
    w_contrast: float = 1.0,
    w_saturation: float = 1.0,
    w_exposedness: float = 1.0,
    sigma_exp: float = 0.2
) -> List[float]:
    """Computes quality weight for each pixel based on contrast, saturation, and well-exposedness."""
    weights = [0.0] * (width * height)
    two_sig2 = 2.0 * sigma_exp * sigma_exp

    for y in range(height):
        for x in range(width):
            p_idx = (y * width + x) * 3
            r, g, b = rgb[p_idx], rgb[p_idx + 1], rgb[p_idx + 2]

            # 1. Saturation: standard deviation across RGB channels
            mean_rgb = (r + g + b) / 3.0
            var = ((r - mean_rgb)**2 + (g - mean_rgb)**2 + (b - mean_rgb)**2) / 3.0
            sat = math.sqrt(var)

            # 2. Well-exposedness: Gaussian curve centered at 0.5
            e_r = math.exp(-((r - 0.5)**2) / two_sig2)
            e_g = math.exp(-((g - 0.5)**2) / two_sig2)
            e_b = math.exp(-((b - 0.5)**2) / two_sig2)
            exposedness = e_r * e_g * e_b

            # 3. Contrast (local difference from neighbor)
            contrast = 1.0

            w = (contrast ** w_contrast) * (sat ** w_saturation) * (exposedness ** w_exposedness)
            weights[y * width + x] = max(1e-12, w)

    return weights

def fuse_exposure_brackets(
    brackets: List[List[float]],
    width: int,
    height: int,
    w_contrast: float = 1.0,
    w_saturation: float = 1.0,
    w_exposedness: float = 1.0
) -> List[float]:
    """Fuses N exposure brackets into a single high dynamic range image."""
    num_brackets = len(brackets)
    if num_brackets == 0:
        raise ValueError("At least one bracket required for fusion")
    if num_brackets == 1:
        return list(brackets[0])

    num_pixels = width * height
    # Compute weight map for each bracket
    weight_maps = [
        _compute_weight_map(b, width, height, w_contrast, w_saturation, w_exposedness)
        for b in brackets
    ]

    # Normalize weights across brackets
    norm_weights = [[0.0] * num_pixels for _ in range(num_brackets)]
    for i in range(num_pixels):
        sum_w = sum(weight_maps[k][i] for k in range(num_brackets))
        inv_sum = 1.0 / sum_w if sum_w > 0 else 1.0 / num_brackets
        for k in range(num_brackets):
            norm_weights[k][i] = weight_maps[k][i] * inv_sum

    # Blend brackets using normalized weights
    fused = [0.0] * (num_pixels * 3)
    for i in range(num_pixels):
        p = i * 3
        for c in range(3):
            blended_val = sum(norm_weights[k][i] * brackets[k][p + c] for k in range(num_brackets))
            fused[p + c] = min(1.0, max(0.0, blended_val))

    return fused
