"""
Tone mapping operators: Reinhard Global, Drago Adaptive Logarithmic, and ACES Filmic curve.
"""
import math
from typing import List

def tone_map_reinhard(
    rgb: List[float],
    key: float = 0.18,
    white_point: float = 4.0
) -> List[float]:
    """Reinhard tone reproduction with high-luminance white point burnout control."""
    # 1. Compute log-average geometric mean luminance
    eps = 1e-6
    log_sum = 0.0
    count = len(rgb) // 3

    for i in range(0, len(rgb), 3):
        lum = 0.2126 * rgb[i] + 0.7152 * rgb[i + 1] + 0.0722 * rgb[i + 2]
        log_sum += math.log(max(eps, lum))

    log_avg = math.exp(log_sum / count)
    scale = key / max(eps, log_avg)
    white_pt2 = white_point * white_point

    out = [0.0] * len(rgb)
    for i in range(0, len(rgb), 3):
        r, g, b = rgb[i], rgb[i + 1], rgb[i + 2]
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        l_scaled = lum * scale

        # L_d = (l_scaled * (1.0 + l_scaled / white_pt2)) / (1.0 + l_scaled)
        l_d = (l_scaled * (1.0 + l_scaled / white_pt2)) / (1.0 + l_scaled)
        lum_ratio = l_d / max(eps, lum)

        out[i] = min(1.0, max(0.0, r * lum_ratio))
        out[i + 1] = min(1.0, max(0.0, g * lum_ratio))
        out[i + 2] = min(1.0, max(0.0, b * lum_ratio))

    return out

def tone_map_drago(
    rgb: List[float],
    bias: float = 0.85
) -> List[float]:
    """Adaptive logarithmic tone mapping preserving high-contrast night and day scenes."""
    eps = 1e-6
    max_lum = 0.0
    for i in range(0, len(rgb), 3):
        lum = 0.2126 * rgb[i] + 0.7152 * rgb[i + 1] + 0.0722 * rgb[i + 2]
        if lum > max_lum:
            max_lum = lum

    if max_lum <= eps:
        return list(rgb)

    log_max_lum = math.log10(max_lum + 1.0)
    bias_power = math.log(bias) / math.log(0.5)

    out = [0.0] * len(rgb)
    for i in range(0, len(rgb), 3):
        r, g, b = rgb[i], rgb[i + 1], rgb[i + 2]
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        norm_lum = lum / max_lum
        denom = math.log(2.0 + 8.0 * (norm_lum ** bias_power))
        l_d = (math.log(lum + 1.0) / denom) / log_max_lum
        ratio = l_d / max(eps, lum)

        out[i] = min(1.0, max(0.0, r * ratio))
        out[i + 1] = min(1.0, max(0.0, g * ratio))
        out[i + 2] = min(1.0, max(0.0, b * ratio))

    return out

def tone_map_aces(rgb: List[float]) -> List[float]:
    """ACES Filmic tone curve fit (Narkowicz 2015).

    f(x) = (x * (a*x + b)) / (x * (c*x + d) + e)
    where a=2.51, b=0.03, c=2.43, d=0.59, e=0.14
    """
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    out = [0.0] * len(rgb)

    for i, v in enumerate(rgb):
        x = max(0.0, v)
        val = (x * (a * x + b)) / (x * (c * x + d) + e)
        out[i] = min(1.0, max(0.0, val))

    return out
