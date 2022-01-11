"""
Color processing stages: white balance, color correction matrices, and electro-optical transfer functions.
"""
from typing import List, Tuple

# Standard Color Correction Matrix (Sensor RGB to sRGB D65)
DEFAULT_CCM_SRGB = [
    1.6437, -0.6302, -0.0135,
    -0.2607,  1.4429, -0.1822,
    -0.0357, -0.3789,  1.4146
]

def apply_white_balance(
    rgb: List[float],
    gain_r: float = 1.0,
    gain_g: float = 1.0,
    gain_b: float = 1.0
) -> List[float]:
    """Applies von Kries diagonal channel gains to compensate for illuminant color temperature."""
    out = [0.0] * len(rgb)
    for i in range(0, len(rgb), 3):
        out[i] = min(1.0, max(0.0, rgb[i] * gain_r))
        out[i + 1] = min(1.0, max(0.0, rgb[i + 1] * gain_g))
        out[i + 2] = min(1.0, max(0.0, rgb[i + 2] * gain_b))
    return out

def estimate_white_balance_gray_world(rgb: List[float]) -> Tuple[float, float, float]:
    """Estimates neutral white balance gains using the Gray World assumption."""
    sum_r, sum_g, sum_b = 0.0, 0.0, 0.0
    count = len(rgb) // 3
    if count == 0:
        return 1.0, 1.0, 1.0

    for i in range(0, len(rgb), 3):
        sum_r += rgb[i]
        sum_g += rgb[i + 1]
        sum_b += rgb[i + 2]

    avg_r = sum_r / count
    avg_g = sum_g / count
    avg_b = sum_b / count

    gain_r = avg_g / avg_r if avg_r > 1e-6 else 1.0
    gain_g = 1.0
    gain_b = avg_g / avg_b if avg_b > 1e-6 else 1.0

    return gain_r, gain_g, gain_b

def apply_color_correction_matrix(
    rgb: List[float],
    ccm: List[float] = DEFAULT_CCM_SRGB
) -> List[float]:
    """Applies 3x3 spectral cross-talk color correction matrix.

    [R_out]   [m00 m01 m02] [R_in]
    [G_out] = [m10 m11 m12] [G_in]
    [B_out]   [m20 m21 m22] [B_in]
    """
    m00, m01, m02 = ccm[0], ccm[1], ccm[2]
    m10, m11, m12 = ccm[3], ccm[4], ccm[5]
    m20, m21, m22 = ccm[6], ccm[7], ccm[8]

    out = [0.0] * len(rgb)
    for i in range(0, len(rgb), 3):
        r, g, b = rgb[i], rgb[i + 1], rgb[i + 2]
        r_out = m00 * r + m01 * g + m02 * b
        g_out = m10 * r + m11 * g + m12 * b
        b_out = m20 * r + m21 * g + m22 * b

        out[i] = min(1.0, max(0.0, r_out))
        out[i + 1] = min(1.0, max(0.0, g_out))
        out[i + 2] = min(1.0, max(0.0, b_out))
    return out

def apply_gamma_curve(rgb: List[float], gamma: float = 2.2) -> List[float]:
    """Applies simple power law gamma curve: V_out = V_in^(1 / gamma)."""
    inv_gamma = 1.0 / gamma if gamma > 0 else 1.0
    out = [0.0] * len(rgb)
    for i, v in enumerate(rgb):
        out[i] = min(1.0, max(0.0, v)) ** inv_gamma
    return out

def apply_srgb_eotf(rgb: List[float]) -> List[float]:
    """Applies the precise IEC 61966-2-1 standard piecewise sRGB transfer function."""
    out = [0.0] * len(rgb)
    for i, v in enumerate(rgb):
        c = min(1.0, max(0.0, v))
        if c <= 0.0031308:
            val = 12.92 * c
        else:
            val = 1.055 * (c ** (1.0 / 2.4)) - 0.055
        out[i] = min(1.0, max(0.0, val))
    return out
