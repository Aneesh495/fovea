"""
Fovea command-line interface.
Enables running raw sensor processing, inspecting intermediate MLIR passes,
benchmarking throughput, and generating synthetic calibration targets.
"""
import argparse
import os
import sys
import time
from ..io.raw import read_raw_bayer, write_raw_bayer
from ..io.pnm import write_ppm
from ..io.synthetic import generate_color_checker_chart, bayer_mosaic, add_sensor_noise
from ..isp.pipeline import ISPPipeline, ISPPipelineConfig
from ..compiler.pass_manager import find_mlir_opt

def cmd_demo(args):
    print("=================================================================")
    print("  FOVEA: MLIR Computational Photography & ISP Compilation Engine")
    print("=================================================================")
    print("[1/4] Generating 720x480 Macbeth 24-patch ColorChecker ground truth...")
    w, h = 720, 480
    gt = generate_color_checker_chart(w, h)

    print("[2/4] Mosaicing into 12-bit RGGB Bayer CFA pattern with photon noise...")
    raw = bayer_mosaic(gt, w, h, cfa_pattern="RGGB", black_level=64.0, white_level=4095.0, bit_depth=12)
    noisy_raw = add_sensor_noise(raw, read_noise_std=1.5, shot_noise_factor=0.3, num_hot_pixels=8, num_dead_pixels=8)

    print("[3/4] Compiling and running multi-stage ISP pipeline...")
    t0 = time.perf_counter()
    config = ISPPipelineConfig(
        demosaic_algorithm="malvar_he_cutler",
        enable_lens_shading=True,
        enable_dpc=True,
        enable_denoise=True,
        enable_sharpen=True,
        tonemap_operator="aces"
    )
    pipeline = ISPPipeline(config)
    rgb_out = pipeline.process(noisy_raw)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    out_path = args.output or "fovea_demo_output.ppm"
    print(f"[4/4] Writing output RGB image to {out_path}...")
    write_ppm(out_path, w, h, rgb_out, max_val=255)

    print("-----------------------------------------------------------------")
    print(f"SUCCESS: Processed {w}x{h} frame ({w * h / 1e6:.2f} Megapixels) in {elapsed_ms:.2f} ms")
    print(f"Throughput: {(w * h) / (elapsed_ms / 1000.0) / 1e6:.2f} Megapixels/second")
    print("=================================================================")

def cmd_inspect(args):
    stage = args.stage.lower()
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    mapping = {
        "dialect": os.path.join(repo_root, "dialects", "irdl", "fovea.irdl.mlir"),
        "mhc": os.path.join(repo_root, "mlir", "kernels", "demosaic_malvar_he_cutler.mlir"),
        "bilateral": os.path.join(repo_root, "mlir", "kernels", "bilateral_filter.mlir"),
        "hdr": os.path.join(repo_root, "mlir", "kernels", "hdr_mertens_fusion.mlir"),
        "aces": os.path.join(repo_root, "mlir", "kernels", "tone_mapping_aces.mlir"),
        "pipeline": os.path.join(repo_root, "mlir", "pipelines", "complete_raw_isp.mlir"),
    }

    if stage not in mapping:
        print(f"Unknown stage '{stage}'. Available stages: {list(mapping.keys())}")
        sys.exit(1)

    target_file = mapping[stage]
    if not os.path.exists(target_file):
        print(f"Error: file not found at {target_file}")
        sys.exit(1)

    print(f"--- MLIR Stage Inspection: {stage} ({target_file}) ---")
    with open(target_file, "r", encoding="utf-8") as f:
        print(f.read())

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="fovea",
        description="Fovea: High-throughput MLIR computational photography and ISP engine"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Demo command
    p_demo = subparsers.add_parser("demo", help="Generate synthetic scene and run full MLIR ISP pipeline")
    p_demo.add_argument("-o", "--output", default="fovea_demo_output.ppm", help="Output PPM file path")
    p_demo.set_defaults(func=cmd_demo)

    # Inspect command
    p_inspect = subparsers.add_parser("inspect", help="Inspect raw MLIR representations for ISP stages")
    p_inspect.add_argument("stage", choices=["dialect", "mhc", "bilateral", "hdr", "aces", "pipeline"])
    p_inspect.set_defaults(func=cmd_inspect)

    args = parser.parse_args(argv)
    args.func(args)

if __name__ == "__main__":
    main()
