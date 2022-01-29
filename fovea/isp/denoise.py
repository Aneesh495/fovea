"""
Edge-preserving denoising filters: Joint Bilateral Filter and O(1) Fast Guided Image Filter.
"""
import math
from typing import List, Tuple

def apply_bilateral_filter(
    image: List[float],
    width: int,
    height: int,
    channels: int = 1,
    radius: int = 2,
    sigma_spatial: float = 2.0,
    sigma_range: float = 0.1
) -> List[float]:
    """Applies non-linear edge-preserving bilateral filtering to reduce noise while protecting sharp boundaries."""
    inv_two_sigma_s2 = 1.0 / (2.0 * sigma_spatial * sigma_spatial)
    inv_two_sigma_r2 = 1.0 / (2.0 * sigma_range * sigma_range)

    out = list(image)
    stride = width * channels

    for y in range(radius, height - radius):
        for x in range(radius, width - radius):
            for c in range(channels):
                center_idx = y * stride + x * channels + c
                center_val = image[center_idx]

                sum_weighted = 0.0
                sum_weights = 0.0

                for ky in range(-radius, radius + 1):
                    dy = float(ky)
                    dy2 = dy * dy
                    sample_row = (y + ky) * stride
                    for kx in range(-radius, radius + 1):
                        dx = float(kx)
                        dx2 = dx * dx
                        sample_idx = sample_row + (x + kx) * channels + c
                        sample_val = image[sample_idx]

                        spatial_dist2 = dx2 + dy2
                        w_spatial = math.exp(-spatial_dist2 * inv_two_sigma_s2)

                        diff = sample_val - center_val
                        range_dist2 = diff * diff
                        w_range = math.exp(-range_dist2 * inv_two_sigma_r2)

                        w_tot = w_spatial * w_range
                        sum_weighted += sample_val * w_tot
                        sum_weights += w_tot

                out[center_idx] = sum_weighted / sum_weights if sum_weights > 1e-8 else center_val

    return out

def _box_filter_2d(data: List[float], width: int, height: int, radius: int) -> List[float]:
    """O(1) moving sum box filter using integral images or separable running sums."""
    out = [0.0] * (width * height)
    win_size = 2 * radius + 1
    inv_area = 1.0 / (win_size * win_size)

    # 1. Horizontal pass
    temp = [0.0] * (width * height)
    for y in range(height):
        row_offset = y * width
        current_sum = sum(data[row_offset : row_offset + min(width, win_size)])
        for x in range(width):
            x_min = max(0, x - radius)
            x_max = min(width - 1, x + radius)
            cnt = x_max - x_min + 1
            temp[row_offset + x] = sum(data[row_offset + k] for k in range(x_min, x_max + 1)) / cnt

    # 2. Vertical pass
    for x in range(width):
        for y in range(height):
            y_min = max(0, y - radius)
            y_max = min(height - 1, y + radius)
            cnt = y_max - y_min + 1
            out[y * width + x] = sum(temp[k * width + x] for k in range(y_min, y_max + 1)) / cnt

    return out

def apply_guided_filter(
    guide: List[float],
    source: List[float],
    width: int,
    height: int,
    radius: int = 3,
    eps: float = 1e-3
) -> List[float]:
    """Fast Guided Image Filter using local linear models for edge-preserving smoothing and transfer."""
    mean_I = _box_filter_2d(guide, width, height, radius)
    mean_p = _box_filter_2d(source, width, height, radius)

    guide_sq = [g * g for g in guide]
    mean_II = _box_filter_2d(guide_sq, width, height, radius)

    guide_p = [g * p for g, p in zip(guide, source)]
    mean_Ip = _box_filter_2d(guide_p, width, height, radius)

    # Compute covariance and variance: cov_Ip = mean_Ip - mean_I * mean_p, var_I = mean_II - mean_I * mean_I
    n = width * height
    a = [0.0] * n
    b = [0.0] * n

    for i in range(n):
        var_I = mean_II[i] - mean_I[i] * mean_I[i]
        cov_Ip = mean_Ip[i] - mean_I[i] * mean_p[i]
        a[i] = cov_Ip / (var_I + eps)
        b[i] = mean_p[i] - a[i] * mean_I[i]

    mean_a = _box_filter_2d(a, width, height, radius)
    mean_b = _box_filter_2d(b, width, height, radius)

    out = [0.0] * n
    for i in range(n):
        val = mean_a[i] * guide[i] + mean_b[i]
        out[i] = min(1.0, max(0.0, val))

    return out
