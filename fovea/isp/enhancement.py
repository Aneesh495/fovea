"""
Detail enhancement and unsharp masking algorithms.
"""
from typing import List

def _separable_gaussian_blur_2d(
    image: List[float],
    width: int,
    height: int,
    radius: int = 2
) -> List[float]:
    """Applies separable 5-tap binomial Gaussian filter [1, 4, 6, 4, 1] / 16."""
    weights = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0]

    # Horizontal pass
    temp = [0.0] * (width * height)
    for y in range(height):
        y_off = y * width
        for x in range(width):
            acc = 0.0
            for kx in range(-2, 3):
                sample_x = min(width - 1, max(0, x + kx))
                acc += image[y_off + sample_x] * weights[kx + 2]
            temp[y_off + x] = acc

    # Vertical pass
    out = [0.0] * (width * height)
    for x in range(width):
        for y in range(height):
            acc = 0.0
            for ky in range(-2, 3):
                sample_y = min(height - 1, max(0, y + ky))
                acc += temp[sample_y * width + x] * weights[ky + 2]
            out[y * width + x] = acc

    return out

def apply_unsharp_mask(
    image: List[float],
    width: int,
    height: int,
    channels: int = 1,
    strength: float = 0.5,
    threshold: float = 0.02
) -> List[float]:
    """Applies unsharp masking to boost edge contrast while suppressing low-frequency halos and noise."""
    out = [0.0] * len(image)
    stride = width * channels

    for c in range(channels):
        channel_data = [image[y * stride + x * channels + c] for y in range(height) for x in range(width)]
        blurred = _separable_gaussian_blur_2d(channel_data, width, height)

        for i in range(len(channel_data)):
            orig = channel_data[i]
            diff = orig - blurred[i]
            if abs(diff) >= threshold:
                val = orig + strength * diff
            else:
                val = orig

            # Interleave back
            y = i // width
            x = i % width
            out[y * stride + x * channels + c] = min(1.0, max(0.0, val))

    return out
