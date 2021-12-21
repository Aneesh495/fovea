"""
Color Filter Array (CFA) demosaicing algorithms.
Implements Bilinear, Malvar-He-Cutler (5x5 optimal FIR), and Directional Gradient demosaicing.
"""
from typing import List, Tuple

def demosaic_bilinear(
    raw: List[float],
    width: int,
    height: int,
    cfa_pattern: str = "RGGB"
) -> List[float]:
    """Bilinear Bayer demosaicing into interleaved RGB float buffer."""
    rgb = [0.0] * (width * height * 3)

    for y in range(1, height - 1):
        y_even = (y % 2 == 0)
        for x in range(1, width - 1):
            x_even = (x % 2 == 0)
            idx = y * width + x
            center = raw[idx]

            top = raw[(y - 1) * width + x]
            bot = raw[(y + 1) * width + x]
            left = raw[y * width + (x - 1)]
            right = raw[y * width + (x + 1)]

            tl = raw[(y - 1) * width + (x - 1)]
            tr = raw[(y - 1) * width + (x + 1)]
            bl = raw[(y + 1) * width + (x - 1)]
            br = raw[(y + 1) * width + (x + 1)]

            cross_avg = (top + bot + left + right) * 0.25
            diag_avg = (tl + tr + bl + br) * 0.25
            tb_avg = (top + bot) * 0.5
            lr_avg = (left + right) * 0.5

            if cfa_pattern == "RGGB":
                if y_even and x_even:
                    r, g, b = center, cross_avg, diag_avg
                elif y_even and not x_even:
                    r, g, b = lr_avg, center, tb_avg
                elif not y_even and x_even:
                    r, g, b = tb_avg, center, lr_avg
                else:
                    r, g, b = diag_avg, cross_avg, center
            elif cfa_pattern == "BGGR":
                if y_even and x_even:
                    b, g, r = center, cross_avg, diag_avg
                elif y_even and not x_even:
                    b, g, r = lr_avg, center, tb_avg
                elif not y_even and x_even:
                    b, g, r = tb_avg, center, lr_avg
                else:
                    b, g, r = diag_avg, cross_avg, center
            else:
                # Default fallback
                r, g, b = center, center, center

            p = idx * 3
            rgb[p] = r
            rgb[p + 1] = g
            rgb[p + 2] = b

    return rgb

def demosaic_malvar_he_cutler(
    raw: List[float],
    width: int,
    height: int,
    cfa_pattern: str = "RGGB"
) -> List[float]:
    """Malvar-He-Cutler (MHC) 5x5 high-quality linear FIR demosaicing.

    Uses optimal 5x5 spatial filters with cross-channel Laplacian guidance
    to reconstruct missing channels without zippering or false color artifacts.
    """
    rgb = [0.0] * (width * height * 3)

    for y in range(2, height - 2):
        y_even = (y % 2 == 0)
        for x in range(2, width - 2):
            x_even = (x % 2 == 0)
            idx = y * width + x
            center = raw[idx]

            # 5x5 cross neighbors
            top1 = raw[(y - 1) * width + x]
            bot1 = raw[(y + 1) * width + x]
            left1 = raw[y * width + (x - 1)]
            right1 = raw[y * width + (x + 1)]

            top2 = raw[(y - 2) * width + x]
            bot2 = raw[(y + 2) * width + x]
            left2 = raw[y * width + (x - 2)]
            right2 = raw[y * width + (x + 2)]

            tl = raw[(y - 1) * width + (x - 1)]
            tr = raw[(y - 1) * width + (x + 1)]
            bl = raw[(y + 1) * width + (x - 1)]
            br = raw[(y + 1) * width + (x + 1)]

            if cfa_pattern == "RGGB":
                if y_even and x_even:
                    # R at R
                    r = center
                    # G at R: 4*center + 2*(T1+B1+L1+R1) - (T2+B2+L2+R2) / 8
                    g = (4.0 * center + 2.0 * (top1 + bot1 + left1 + right1) - (top2 + bot2 + left2 + right2)) / 8.0
                    # B at R: 6*center + 2*(TL+TR+BL+BR) - 1.5*(T2+B2+L2+R2) / 8
                    b = (6.0 * center + 2.0 * (tl + tr + bl + br) - 1.5 * (top2 + bot2 + left2 + right2)) / 8.0
                elif y_even and not x_even:
                    # Gr
                    g = center
                    # R at Gr: 5*center + 4*(L1+R1) - (L2+R2) - (TL+TR+BL+BR) + 0.5*(T2+B2) / 8
                    r = (5.0 * center + 4.0 * (left1 + right1) - (left2 + right2) - (tl + tr + bl + br) + 0.5 * (top2 + bot2)) / 8.0
                    # B at Gr: 5*center + 4*(T1+B1) - (T2+B2) - (TL+TR+BL+BR) + 0.5*(L2+R2) / 8
                    b = (5.0 * center + 4.0 * (top1 + bot1) - (top2 + bot2) - (tl + tr + bl + br) + 0.5 * (left2 + right2)) / 8.0
                elif not y_even and x_even:
                    # Gb
                    g = center
                    # R at Gb
                    r = (5.0 * center + 4.0 * (top1 + bot1) - (top2 + bot2) - (tl + tr + bl + br) + 0.5 * (left2 + right2)) / 8.0
                    # B at Gb
                    b = (5.0 * center + 4.0 * (left1 + right1) - (left2 + right2) - (tl + tr + bl + br) + 0.5 * (top2 + bot2)) / 8.0
                else:
                    # B at B
                    b = center
                    # G at B
                    g = (4.0 * center + 2.0 * (top1 + bot1 + left1 + right1) - (top2 + bot2 + left2 + right2)) / 8.0
                    # R at B
                    r = (6.0 * center + 2.0 * (tl + tr + bl + br) - 1.5 * (top2 + bot2 + left2 + right2)) / 8.0
            else:
                r, g, b = center, center, center

            p = idx * 3
            rgb[p] = min(1.0, max(0.0, r))
            rgb[p + 1] = min(1.0, max(0.0, g))
            rgb[p + 2] = min(1.0, max(0.0, b))

    return rgb

def demosaic_directional(
    raw: List[float],
    width: int,
    height: int,
    cfa_pattern: str = "RGGB"
) -> List[float]:
    """Adaptive edge-directed gradient demosaicing.

    Calculates horizontal and vertical second-order spatial gradients.
    Interpolates along the direction of minimum gradient to preserve sharp edges.
    """
    rgb = [0.0] * (width * height * 3)

    for y in range(2, height - 2):
        y_even = (y % 2 == 0)
        for x in range(2, width - 2):
            x_even = (x % 2 == 0)
            idx = y * width + x
            center = raw[idx]

            top1 = raw[(y - 1) * width + x]
            bot1 = raw[(y + 1) * width + x]
            left1 = raw[y * width + (x - 1)]
            right1 = raw[y * width + (x + 1)]

            top2 = raw[(y - 2) * width + x]
            bot2 = raw[(y + 2) * width + x]
            left2 = raw[y * width + (x - 2)]
            right2 = raw[y * width + (x + 2)]

            # Gradients
            grad_h = abs(left1 - right1) + abs(2.0 * center - left2 - right2)
            grad_v = abs(top1 - bot1) + abs(2.0 * center - top2 - bot2)

            if grad_h < grad_v:
                # Interpolate horizontally
                g_interp = 0.5 * (left1 + right1) + 0.25 * (2.0 * center - left2 - right2)
            elif grad_v < grad_h:
                # Interpolate vertically
                g_interp = 0.5 * (top1 + bot1) + 0.25 * (2.0 * center - top2 - bot2)
            else:
                # Average
                g_interp = 0.25 * (top1 + bot1 + left1 + right1)

            if y_even and x_even:
                r = center
                g = g_interp
                b = 0.25 * (raw[(y - 1) * width + (x - 1)] + raw[(y - 1) * width + (x + 1)] +
                            raw[(y + 1) * width + (x - 1)] + raw[(y + 1) * width + (x + 1)])
            elif not y_even and not x_even:
                b = center
                g = g_interp
                r = 0.25 * (raw[(y - 1) * width + (x - 1)] + raw[(y - 1) * width + (x + 1)] +
                            raw[(y + 1) * width + (x - 1)] + raw[(y + 1) * width + (x + 1)])
            else:
                g = center
                r = 0.5 * (left1 + right1) if y_even else 0.5 * (top1 + bot1)
                b = 0.5 * (top1 + bot1) if y_even else 0.5 * (left1 + right1)

            p = idx * 3
            rgb[p] = min(1.0, max(0.0, r))
            rgb[p + 1] = min(1.0, max(0.0, g))
            rgb[p + 2] = min(1.0, max(0.0, b))

    return rgb
