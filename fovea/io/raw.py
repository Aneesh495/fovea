"""
Raw Bayer sensor image container and zero-dependency binary serializer.
Handles 10-bit, 12-bit, 14-bit, and 16-bit raw mosaic frames with custom metadata.
"""
import struct
from typing import List, Optional, Tuple

MAGIC_RAW = b"FOVEARAW"

class RawBayerImage:
    """Container for raw single-channel Bayer mosaic sensor data."""

    def __init__(
        self,
        width: int,
        height: int,
        cfa_pattern: str = "RGGB",
        bit_depth: int = 12,
        black_level: float = 64.0,
        white_level: float = 4095.0,
        data: Optional[List[float]] = None
    ):
        if cfa_pattern not in ("RGGB", "BGGR", "GBRG", "GRBG"):
            raise ValueError(f"Unsupported CFA pattern: {cfa_pattern}")
        if bit_depth not in (8, 10, 12, 14, 16):
            raise ValueError(f"Unsupported bit depth: {bit_depth}")
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions: {width}x{height}")

        self.width = width
        self.height = height
        self.cfa_pattern = cfa_pattern
        self.bit_depth = bit_depth
        self.black_level = black_level
        self.white_level = white_level
        self.data: List[float] = data if data is not None else [0.0] * (width * height)

    def get_pixel(self, x: int, y: int) -> float:
        return self.data[y * self.width + x]

    def set_pixel(self, x: int, y: int, value: float):
        self.data[y * self.width + x] = value

def write_raw_bayer(image: RawBayerImage, filepath: str):
    """Writes raw Bayer mosaic into a compact binary format with a 32-byte header."""
    header = bytearray(32)
    header[0:8] = MAGIC_RAW
    # Format: width(u32), height(u32), bit_depth(u16), cfa(4s), black_level(f32), white_level(f32)
    cfa_bytes = image.cfa_pattern.encode("ascii").ljust(4, b"\x00")
    payload_header = struct.pack(
        "<IIH4sff",
        image.width,
        image.height,
        image.bit_depth,
        cfa_bytes,
        image.black_level,
        image.white_level
    )
    header[8:8+len(payload_header)] = payload_header

    # Pack pixel data as 16-bit unsigned integers
    max_val = (1 << image.bit_depth) - 1
    raw_shorts = [min(max(0, int(round(v))), max_val) for v in image.data]
    pixel_bytes = struct.pack(f"<{len(raw_shorts)}H", *raw_shorts)

    with open(filepath, "wb") as f:
        f.write(header)
        f.write(pixel_bytes)

def read_raw_bayer(filepath: str) -> RawBayerImage:
    """Reads raw Bayer mosaic from a binary file."""
    with open(filepath, "rb") as f:
        header = f.read(32)
        if len(header) < 32 or header[0:8] != MAGIC_RAW:
            raise ValueError("Invalid Fovea RAW magic header")

        width, height, bit_depth, cfa_bytes, black_level, white_level = struct.unpack_from(
            "<IIH4sff", header, 8
        )
        cfa_pattern = cfa_bytes.decode("ascii").rstrip("\x00")

        expected_pixels = width * height
        pixel_bytes = f.read(expected_pixels * 2)
        if len(pixel_bytes) < expected_pixels * 2:
            raise ValueError("Truncated raw pixel payload")

        raw_shorts = struct.unpack(f"<{expected_pixels}H", pixel_bytes)
        data = [float(v) for v in raw_shorts]

        return RawBayerImage(
            width=width,
            height=height,
            cfa_pattern=cfa_pattern,
            bit_depth=bit_depth,
            black_level=black_level,
            white_level=white_level,
            data=data
        )
