"""
Multi-scale Gaussian and Laplacian pyramids for multi-resolution fusion and tone compression.
"""
from typing import List, Tuple

def _pyr_down_single(
    image: List[float],
    width: int,
    height: int
) -> Tuple[List[float], int, int]:
    """Downsamples image by 2x using 5-tap separable binomial filter."""
    weights = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0]
    out_w = max(1, width // 2)
    out_h = max(1, height // 2)

    # 1. Horizontal blur and decimate
    temp = [0.0] * (out_w * height)
    for y in range(height):
        y_off = y * width
        t_off = y * out_w
        for ox in range(out_w):
            ix = ox * 2
            acc = 0.0
            for k in range(-2, 3):
                sx = min(width - 1, max(0, ix + k))
                acc += image[y_off + sx] * weights[k + 2]
            temp[t_off + ox] = acc

    # 2. Vertical blur and decimate
    out = [0.0] * (out_w * out_h)
    for ox in range(out_w):
        for oy in range(out_h):
            iy = oy * 2
            acc = 0.0
            for k in range(-2, 3):
                sy = min(height - 1, max(0, iy + k))
                acc += temp[sy * out_w + ox] * weights[k + 2]
            out[oy * out_w + ox] = acc

    return out, out_w, out_h

def _pyr_up_single(
    image: List[float],
    width: int,
    height: int,
    target_w: int,
    target_h: int
) -> List[float]:
    """Upsamples image by 2x with bilinear interpolation."""
    out = [0.0] * (target_w * target_h)
    scale_x = float(width) / float(target_w)
    scale_y = float(height) / float(target_h)

    for y in range(target_h):
        src_y = y * scale_y
        y0 = int(src_y)
        y1 = min(height - 1, y0 + 1)
        fy = src_y - y0

        for x in range(target_w):
            src_x = x * scale_x
            x0 = int(src_x)
            x1 = min(width - 1, x0 + 1)
            fx = src_x - x0

            p00 = image[y0 * width + x0]
            p10 = image[y0 * width + x1]
            p01 = image[y1 * width + x0]
            p11 = image[y1 * width + x1]

            val = (1.0 - fx) * (1.0 - fy) * p00 + fx * (1.0 - fy) * p10 + (1.0 - fx) * fy * p01 + fx * fy * p11
            out[y * target_w + x] = val

    return out

def build_gaussian_pyramid(
    image: List[float],
    width: int,
    height: int,
    levels: int = 4
) -> List[Tuple[List[float], int, int]]:
    """Builds a multi-scale Gaussian pyramid [G_0, G_1, ..., G_{levels-1}]."""
    pyramid = [(list(image), width, height)]
    cur_img, cur_w, cur_h = image, width, height

    for _ in range(1, levels):
        if cur_w <= 2 or cur_h <= 2:
            break
        down, cur_w, cur_h = _pyr_down_single(cur_img, cur_w, cur_h)
        pyramid.append((down, cur_w, cur_h))
        cur_img = down

    return pyramid

def build_laplacian_pyramid(
    image: List[float],
    width: int,
    height: int,
    levels: int = 4
) -> List[Tuple[List[float], int, int]]:
    """Builds a multi-scale Laplacian pyramid [L_0, L_1, ..., L_{levels-1}]."""
    gauss = build_gaussian_pyramid(image, width, height, levels)
    laplacian = []

    for i in range(len(gauss) - 1):
        g_cur, w_cur, h_cur = gauss[i]
        g_next, w_next, h_next = gauss[i + 1]

        upsampled = _pyr_up_single(g_next, w_next, h_next, w_cur, h_cur)
        l_cur = [g - u for g, u in zip(g_cur, upsampled)]
        laplacian.append((l_cur, w_cur, h_cur))

    # Top level is the lowest-frequency Gaussian base
    laplacian.append(gauss[-1])
    return laplacian

def reconstruct_laplacian_pyramid(
    laplacian: List[Tuple[List[float], int, int]]
) -> List[float]:
    """Reconstructs full-resolution image from Laplacian pyramid."""
    cur_img, cur_w, cur_h = laplacian[-1]

    for i in reversed(range(len(laplacian) - 1)):
        l_prev, w_prev, h_prev = laplacian[i]
        upsampled = _pyr_up_single(cur_img, cur_w, cur_h, w_prev, h_prev)
        cur_img = [l + u for l, u in zip(l_prev, upsampled)]
        cur_w, cur_h = w_prev, h_prev

    return cur_img
