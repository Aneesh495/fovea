"""
Radiance RGBE (.hdr) high dynamic range format reader and writer.
"""
import math
import struct
from typing import List, Tuple

def _float_to_rgbe(r: float, g: float, b: float) -> Tuple[int, int, int, int]:
    v = max(r, g, b)
    if v < 1e-32:
        return 0, 0, 0, 0
    m, e = math.frexp(v)
    v_norm = m * 256.0 / v
    re = min(255, max(0, int(r * v_norm)))
    ge = min(255, max(0, int(g * v_norm)))
    be = min(255, max(0, int(b * v_norm)))
    ee = min(255, max(0, int(e + 128)))
    return re, ge, be, ee

def _rgbe_to_float(r: int, g: int, b: int, e: int) -> Tuple[float, float, float]:
    if e == 0:
        return 0.0, 0.0, 0.0
    f = math.ldexp(1.0, e - (128 + 8))
    return r * f, g * f, b * f

def write_rgbe_hdr(
    filepath: str,
    width: int,
    height: int,
    rgb_floats: List[float]
):
    """Writes an uncompressed Radiance RGBE (.hdr) file."""
    header = (
        "#?RADIANCE\n"
        "FORMAT=32-bit_rle_rgbe\n"
        "EXPOSURE=1.0\n\n"
        f"-Y {height} +X {width}\n"
    ).encode("ascii")

    pixels = bytearray()
    for i in range(0, len(rgb_floats), 3):
        r, g, b = rgb_floats[i], rgb_floats[i+1], rgb_floats[i+2]
        re, ge, be, ee = _float_to_rgbe(r, g, b)
        pixels.extend((re, ge, be, ee))

    with open(filepath, "wb") as f:
        f.write(header)
        f.write(pixels)

def read_rgbe_hdr(filepath: str) -> Tuple[int, int, List[float]]:
    """Reads basic Radiance RGBE file."""
    with open(filepath, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith(b"-Y"):
                dim_line = line
                break
            header += line

        parts = dim_line.split()
        height = int(parts[1])
        width = int(parts[3])

        expected = width * height
        raw = f.read(expected * 4)
        out = []
        for i in range(0, len(raw), 4):
            re, ge, be, ee = raw[i], raw[i+1], raw[i+2], raw[i+3]
            r, g, b = _rgbe_to_float(re, ge, be, ee)
            out.extend([r, g, b])

        return width, height, out
