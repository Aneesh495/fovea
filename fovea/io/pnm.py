"""
Zero-dependency reader and writer for Netpbm PPM (P6/P3) and PFM (Portable Float Map).
"""
import struct
from typing import List, Tuple

def write_ppm(
    filepath: str,
    width: int,
    height: int,
    rgb_data: List[float],
    max_val: int = 255
):
    """Writes 8-bit or 16-bit binary PPM (P6) file."""
    header = f"P6\n{width} {height}\n{max_val}\n".encode("ascii")
    if max_val <= 255:
        clamped = [min(max(0, int(round(v))), max_val) for v in rgb_data]
        body = bytes(clamped)
    else:
        clamped = [min(max(0, int(round(v))), max_val) for v in rgb_data]
        body = struct.pack(f">{len(clamped)}H", *clamped)

    with open(filepath, "wb") as f:
        f.write(header)
        f.write(body)

def read_ppm(filepath: str) -> Tuple[int, int, List[float]]:
    """Reads binary PPM (P6) file into float RGB buffer."""
    with open(filepath, "rb") as f:
        magic = f.readline().strip()
        if magic != b"P6":
            raise ValueError(f"Only binary P6 PPM supported, got {magic}")

        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()

        parts = line.split()
        while len(parts) < 2:
            parts.extend(f.readline().split())
        width = int(parts[0])
        height = int(parts[1])

        max_line = f.readline()
        while max_line.startswith(b"#"):
            max_line = f.readline()
        max_val = int(max_line.strip())

        expected = width * height * 3
        if max_val <= 255:
            raw = f.read(expected)
            data = [float(b) for b in raw]
        else:
            raw = f.read(expected * 2)
            data = [float(v) for v in struct.unpack(f">{expected}H", raw)]

        return width, height, data

def write_pfm(
    filepath: str,
    width: int,
    height: int,
    rgb_floats: List[float],
    scale: float = -1.0
):
    """Writes 32-bit floating point PFM (Portable Float Map) file for HDR imaging."""
    # "PF" for 3-channel color, negative scale indicates little-endian IEEE-754 floats
    header = f"PF\n{width} {height}\n{scale}\n".encode("ascii")
    
    # PFM stores rows from bottom to top, so invert scanlines
    inverted_data = []
    stride = width * 3
    for y in reversed(range(height)):
        row = rgb_floats[y * stride : (y + 1) * stride]
        inverted_data.extend(row)

    body = struct.pack(f"<{len(inverted_data)}f", *inverted_data)
    with open(filepath, "wb") as f:
        f.write(header)
        f.write(body)

def read_pfm(filepath: str) -> Tuple[int, int, List[float]]:
    """Reads 32-bit floating point PFM file."""
    with open(filepath, "rb") as f:
        line1 = f.readline().strip()
        if line1 != b"PF":
            raise ValueError("Only 3-channel color PFM supported")

        line2 = f.readline()
        while line2.startswith(b"#"):
            line2 = f.readline()
        dims = line2.split()
        width = int(dims[0])
        height = int(dims[1])

        line3 = f.readline().strip()
        scale = float(line3)
        endian = "<" if scale < 0 else ">"

        expected = width * height * 3
        raw = f.read(expected * 4)
        floats = list(struct.unpack(f"{endian}{expected}f", raw))

        # Invert scanlines back to top-to-bottom
        out_data = [0.0] * expected
        stride = width * 3
        for y in range(height):
            inv_y = height - 1 - y
            out_data[y * stride : (y + 1) * stride] = floats[inv_y * stride : (inv_y + 1) * stride]

        return width, height, out_data
