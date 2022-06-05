"""
Throughput benchmarks for Fovea image signal processor stages.
Measures execution time and throughput (Megapixels/sec and FPS) across
calibrated resolutions.
"""

import time
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fovea.io.synthetic import generate_color_checker_chart, bayer_mosaic
from fovea.isp.pipeline import ISPPipeline, ISPPipelineConfig


def benchmark_resolution(width: int, height: int, fast_mode: bool = True, num_runs: int = 3) -> dict:
    """Benchmark the ISP pipeline on a given resolution."""
    chart = generate_color_checker_chart(width=width, height=height)
    mosaic = bayer_mosaic(chart, width=width, height=height, cfa_pattern="RGGB")

    if fast_mode:
        config = ISPPipelineConfig(
            demosaic_algorithm="bilinear",
            enable_denoise=False,
            enable_sharpen=True,
            tonemap_operator="aces",
        )
    else:
        config = ISPPipelineConfig(
            demosaic_algorithm="malvar_he_cutler",
            enable_denoise=False,
            enable_sharpen=True,
            tonemap_operator="aces",
        )

    pipeline = ISPPipeline(config)

    # Warmup
    _ = pipeline.process(mosaic)

    # Timed runs
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = pipeline.process(mosaic)
    elapsed = time.perf_counter() - start

    avg_time = elapsed / num_runs
    fps = 1.0 / avg_time if avg_time > 0 else 0.0
    megapixels = (width * height) / 1e6
    mp_per_sec = megapixels * fps

    return {
        "resolution": f"{width}x{height}",
        "megapixels": megapixels,
        "avg_time_ms": avg_time * 1000.0,
        "fps": fps,
        "throughput_mp_s": mp_per_sec,
    }


def main():
    print("=" * 70)
    print("Fovea ISP Throughput Benchmark (Real-Time Stream Pipeline)")
    print("=" * 70)
    resolutions = [
        (160, 120),
        (320, 240),
        (640, 480),
    ]

    for w, h in resolutions:
        res = benchmark_resolution(w, h, fast_mode=True, num_runs=3)
        print(f"Resolution: {res['resolution']:>10} ({res['megapixels']:.3f} MP) | "
              f"Latency: {res['avg_time_ms']:>8.2f} ms | "
              f"Rate: {res['fps']:>6.2f} FPS | "
              f"Throughput: {res['throughput_mp_s']:>6.2f} MP/s")

    print("\n" + "=" * 70)
    print("Fovea ISP Precision Pipeline (MHC 5x5 Optimal FIR + Sharpening + ACES)")
    print("=" * 70)
    for w, h in [(160, 120), (320, 240)]:
        res = benchmark_resolution(w, h, fast_mode=False, num_runs=2)
        print(f"Resolution: {res['resolution']:>10} ({res['megapixels']:.3f} MP) | "
              f"Latency: {res['avg_time_ms']:>8.2f} ms | "
              f"Rate: {res['fps']:>6.2f} FPS | "
              f"Throughput: {res['throughput_mp_s']:>6.2f} MP/s")

    print("=" * 70)


if __name__ == "__main__":
    main()
