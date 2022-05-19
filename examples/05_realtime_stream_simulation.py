#!/usr/bin/env python3
"""
Example 05: Real-time Camera Stream Frame-by-Frame Simulation.
Measures latency, memory throughput, and frames-per-second across a streaming sequence.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fovea.io.synthetic import generate_color_checker_chart, bayer_mosaic
from fovea.isp.pipeline import ISPPipeline, ISPPipelineConfig

def main():
    print("Fovea Example 05: Real-time Camera Stream Simulation")
    w, h = 320, 240
    num_frames = 10

    gt = generate_color_checker_chart(w, h)
    raw = bayer_mosaic(gt, w, h, "RGGB")

    config = ISPPipelineConfig(
        demosaic_algorithm="bilinear",
        enable_lens_shading=True,
        enable_dpc=False,
        enable_denoise=False,
        enable_sharpen=True,
        tonemap_operator="aces"
    )
    pipeline = ISPPipeline(config)

    print(f"Processing stream of {num_frames} frames ({w}x{h})...")
    latencies = []

    for i in range(num_frames):
        t0 = time.perf_counter()
        rgb = pipeline.process(raw)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

    avg_lat = sum(latencies) / len(latencies)
    fps = 1000.0 / avg_lat if avg_lat > 0 else 0.0
    print(f"Average latency per frame: {avg_lat:.2f} ms")
    print(f"Effective throughput: {fps:.1f} FPS ({(w * h * fps) / 1e6:.2f} Megapixels/sec)")

if __name__ == "__main__":
    main()
