#!/usr/bin/env python3
"""
Quickstart Example: End-to-End Raw Sensor Frame to High-Quality RGB.
Demonstrates sensor linearization, MHC demosaicing, white balance, and ACES tone curve.
"""
import os
import sys

# Ensure fovea is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fovea.io.synthetic import generate_color_checker_chart, bayer_mosaic, add_sensor_noise
from fovea.isp.pipeline import ISPPipeline, ISPPipelineConfig
from fovea.io.pnm import write_ppm

def main():
    print("Fovea Example 01: Raw to RGB Quickstart")
    w, h = 640, 480

    print("Generating synthetic Macbeth ColorChecker target...")
    gt = generate_color_checker_chart(w, h)

    print("Synthesizing 12-bit Bayer RGGB raw sensor frame with physical noise...")
    raw = bayer_mosaic(gt, w, h, cfa_pattern="RGGB", black_level=64.0, white_level=4095.0, bit_depth=12)
    noisy_raw = add_sensor_noise(raw, read_noise_std=1.5, shot_noise_factor=0.2, num_hot_pixels=5, num_dead_pixels=5)

    print("Configuring MLIR-driven ISP pipeline...")
    config = ISPPipelineConfig(
        demosaic_algorithm="malvar_he_cutler",
        enable_lens_shading=True,
        enable_dpc=True,
        enable_denoise=False,
        enable_sharpen=True,
        tonemap_operator="aces"
    )
    pipeline = ISPPipeline(config)

    print("Processing sensor frame...")
    rgb = pipeline.process(noisy_raw)

    out_file = "quickstart_output.ppm"
    write_ppm(out_file, w, h, rgb)
    print(f"Saved processed RGB image to {out_file} ({len(rgb)} values)")

if __name__ == "__main__":
    main()
