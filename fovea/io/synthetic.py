"""
Synthetic calibration target and sensor noise generator for ISP validation.
Generates standard Macbeth 24-patch color charts, Siemens star resolution charts,
sinusoidal zone plates, Bayer CFA mosaicing, and physical sensor noise models.
"""
import math
import random
from typing import List, Tuple
from .raw import RawBayerImage

# Standard 24 Macbeth ColorChecker reflectance values (sRGB reference sRGB D65)
MACBETH_24_SRGB = [
    (115, 82, 68),    # 01. Dark Skin
    (194, 150, 130),  # 02. Light Skin
    (98, 122, 157),   # 03. Blue Sky
    (87, 108, 67),    # 04. Foliage
    (133, 128, 177),  # 05. Blue Flower
    (103, 189, 170),  # 06. Bluish Green
    (214, 126, 44),   # 07. Orange
    (80, 91, 166),    # 08. Purplish Blue
    (193, 90, 99),    # 09. Moderate Red
    (94, 60, 108),    # 10. Purple
    (157, 188, 64),   # 11. Yellow Green
    (224, 163, 46),   # 12. Orange Yellow
    (56, 61, 150),    # 13. Blue
    (70, 148, 73),    # 14. Green
    (175, 54, 60),    # 15. Red
    (231, 199, 31),   # 16. Yellow
    (187, 86, 149),   # 17. Magenta
    (8, 133, 161),    # 18. Cyan
    (243, 243, 242),  # 19. White 9.5
    (200, 200, 200),  # 20. Neutral 8
    (160, 160, 160),  # 21. Neutral 6.5
    (122, 122, 121),  # 22. Neutral 5
    (85, 85, 85),     # 23. Neutral 3.5
    (52, 52, 52),     # 24. Black 2
]

def generate_color_checker_chart(width: int = 720, height: int = 480) -> List[float]:
    """Generates a 6x4 Macbeth ColorChecker test pattern as RGB float array in [0.0, 1.0]."""
    data = [0.0] * (width * height * 3)
    cols, rows = 6, 4
    patch_w = width // cols
    patch_h = height // rows

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            srgb = MACBETH_24_SRGB[idx]
            # Convert 8-bit sRGB to linear float
            rf = (srgb[0] / 255.0) ** 2.2
            gf = (srgb[1] / 255.0) ** 2.2
            bf = (srgb[2] / 255.0) ** 2.2

            x_start = c * patch_w
            x_end = (c + 1) * patch_w if c < cols - 1 else width
            y_start = r * patch_h
            y_end = (r + 1) * patch_h if r < rows - 1 else height

            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    p_idx = (y * width + x) * 3
                    data[p_idx] = rf
                    data[p_idx + 1] = gf
                    data[p_idx + 2] = bf

    return data

def generate_siemens_star(width: int = 512, height: int = 512, spokes: int = 36) -> List[float]:
    """Generates a Siemens star radial resolution chart for optical Nyquist testing."""
    data = [0.0] * (width * height * 3)
    cx = width / 2.0
    cy = height / 2.0
    max_radius = min(cx, cy) * 0.95

    for y in range(height):
        dy = y - cy
        for x in range(width):
            dx = x - cx
            dist = math.hypot(dx, dy)
            if dist > max_radius:
                val = 0.5
            else:
                angle = math.atan2(dy, dx)
                val = 0.5 + 0.5 * math.sin(spokes * angle)
            idx = (y * width + x) * 3
            data[idx] = val
            data[idx + 1] = val
            data[idx + 2] = val

    return data

def bayer_mosaic(
    rgb_data: List[float],
    width: int,
    height: int,
    cfa_pattern: str = "RGGB",
    black_level: float = 64.0,
    white_level: float = 4095.0,
    bit_depth: int = 12
) -> RawBayerImage:
    """Mosaics full-color RGB data into single-channel Bayer sensor frame."""
    bayer_data = [0.0] * (width * height)
    scale = white_level - black_level

    for y in range(height):
        y_even = (y % 2 == 0)
        for x in range(width):
            x_even = (x % 2 == 0)
            p_idx = (y * width + x) * 3
            r, g, b = rgb_data[p_idx], rgb_data[p_idx + 1], rgb_data[p_idx + 2]

            # Determine CFA color channel based on 2x2 grid
            if cfa_pattern == "RGGB":
                channel_val = r if (y_even and x_even) else (b if (not y_even and not x_even) else g)
            elif cfa_pattern == "BGGR":
                channel_val = b if (y_even and x_even) else (r if (not y_even and not x_even) else g)
            elif cfa_pattern == "GRBG":
                channel_val = g if (y_even and x_even) else (r if (y_even and not x_even) else (b if (not y_even and x_even) else g))
            elif cfa_pattern == "GBRG":
                channel_val = g if (y_even and x_even) else (b if (y_even and not x_even) else (r if (not y_even and x_even) else g))
            else:
                raise ValueError(f"Unknown CFA: {cfa_pattern}")

            sensor_val = black_level + channel_val * scale
            bayer_data[y * width + x] = min(max(black_level, sensor_val), white_level)

    return RawBayerImage(
        width=width,
        height=height,
        cfa_pattern=cfa_pattern,
        bit_depth=bit_depth,
        black_level=black_level,
        white_level=white_level,
        data=bayer_data
    )

def add_sensor_noise(
    image: RawBayerImage,
    read_noise_std: float = 2.0,
    shot_noise_factor: float = 0.5,
    num_hot_pixels: int = 10,
    num_dead_pixels: int = 10,
    seed: int = 42
) -> RawBayerImage:
    """Injects photon shot noise, Gaussian read noise, and stuck defective pixels."""
    rng = random.Random(seed)
    noisy_data = list(image.data)
    n = len(noisy_data)

    # 1. Shot noise and Gaussian read noise
    for i in range(n):
        val = noisy_data[i] - image.black_level
        if val > 0:
            shot_std = math.sqrt(val) * shot_noise_factor
            val += rng.gauss(0.0, shot_std)
        read_noise = rng.gauss(0.0, read_noise_std)
        final_val = val + image.black_level + read_noise
        noisy_data[i] = min(max(image.black_level, final_val), image.white_level)

    # 2. Defective hot pixels (saturated stuck high)
    for _ in range(num_hot_pixels):
        idx = rng.randint(0, n - 1)
        noisy_data[idx] = image.white_level

    # 3. Defective dead pixels (stuck at black level)
    for _ in range(num_dead_pixels):
        idx = rng.randint(0, n - 1)
        noisy_data[idx] = image.black_level

    return RawBayerImage(
        width=image.width,
        height=image.height,
        cfa_pattern=image.cfa_pattern,
        bit_depth=image.bit_depth,
        black_level=image.black_level,
        white_level=image.white_level,
        data=noisy_data
    )

def generate_exposure_bracket(
    rgb_data: List[float],
    width: int,
    height: int,
    ev_stops: List[float] = [-2.0, 0.0, 2.0]
) -> List[List[float]]:
    """Generates bracketed exposures for multi-exposure HDR fusion."""
    brackets = []
    for ev in ev_stops:
        scale = 2.0 ** ev
        bracket = [min(1.0, max(0.0, v * scale)) for v in rgb_data]
        brackets.append(bracket)
    return brackets
