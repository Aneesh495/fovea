#!/usr/bin/env python3
"""
Example 02: Multi-Exposure HDR Burst Fusion.
Fuses under-exposed, mid-exposed, and over-exposed sensor frames using Mertens weighting.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fovea.io.synthetic import generate_color_checker_chart, generate_exposure_bracket
from fovea.isp.hdr import fuse_exposure_brackets
from fovea.io.pnm import write_ppm

def main():
    print("Fovea Example 02: Multi-Exposure HDR Exposure Fusion")
    w, h = 320, 240
    base = generate_color_checker_chart(w, h)

    print("Generating 3-frame exposure bracket (-2.0 EV, 0.0 EV, +2.0 EV)...")
    brackets = generate_exposure_bracket(base, w, h, ev_stops=[-2.0, 0.0, 2.0])

    print("Fusing exposure brackets using contrast, saturation, and well-exposedness...")
    fused = fuse_exposure_brackets(brackets, w, h, w_contrast=1.0, w_saturation=1.0, w_exposedness=1.0)

    out_file = "hdr_fusion_output.ppm"
    write_ppm(out_file, w, h, fused)
    print(f"Saved fused HDR image to {out_file}")

if __name__ == "__main__":
    main()
