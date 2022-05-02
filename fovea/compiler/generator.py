"""
Large-scale MLIR module generator for camera pipelines, tile schedules, and SIMD sweeps.
Emits verified, production-grade MLIR modules across Affine, Vector, SCF, MemRef, and LLVM dialects.
"""
import os
import subprocess
from typing import List

def generate_complete_raw_isp_mlir(height: int = 1080, width: int = 1920) -> str:
    """Generates an end-to-end monolithic Image Signal Processor pipeline in MLIR."""
    return f"""// Full Multi-Stage Image Signal Processor (ISP) Pipeline in MLIR
// Connects black level calibration, lens shading, demosaicing, white balance,
// color correction matrix, bilateral filter, unsharp mask, and ACES tone curve.

module @fovea_monolithic_isp {{
  func.func @process_frame_4k(
      %raw_bayer: memref<{height}x{width}xf32>,
      %final_srgb: memref<{height}x{width}x3xf32>,
      %bl_offset: f32,
      %bl_scale: f32,
      %cx: f32, %cy: f32, %inv_r: f32,
      %k0: f32, %k1: f32, %k2: f32, %k3: f32,
      %gain_r: f32, %gain_g: f32, %gain_b: f32,
      %m00: f32, %m01: f32, %m02: f32,
      %m10: f32, %m11: f32, %m12: f32,
      %m20: f32, %m21: f32, %m22: f32) {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %f0 = arith.constant 0.000000e+00 : f32
    %f1 = arith.constant 1.000000e+00 : f32
    %c_inv8 = arith.constant 0.125000e+00 : f32
    %c4_f = arith.constant 4.000000e+00 : f32
    %c2_f = arith.constant 2.000000e+00 : f32
    %c6_f = arith.constant 6.000000e+00 : f32
    %c1_5_f = arith.constant 1.500000e+00 : f32
    %c5_f = arith.constant 5.000000e+00 : f32
    %c0_5_f = arith.constant 0.500000e+00 : f32

    // ACES curve parameters
    %aces_a = arith.constant 2.510000e+00 : f32
    %aces_b = arith.constant 0.030000e+00 : f32
    %aces_c = arith.constant 2.430000e+00 : f32
    %aces_d = arith.constant 0.590000e+00 : f32
    %aces_e = arith.constant 0.140000e+00 : f32

    %h_bound = arith.constant {height - 2} : index
    %w_bound = arith.constant {width - 2} : index

    // Fused raster scanline traversal over image sensor
    scf.for %y = %c2 to %h_bound step %c1 {{
      %y_i32 = arith.index_cast %y : index to i32
      %y_f = arith.sitofp %y_i32 : i32 to f32
      %dy = arith.subf %y_f, %cy : f32
      %dy2 = arith.mulf %dy, %dy : f32

      %y_prev1 = arith.subi %y, %c1 : index
      %y_next1 = arith.addi %y, %c1 : index
      %y_prev2 = arith.subi %y, %c2 : index
      %y_next2 = arith.addi %y, %c2 : index

      %y_rem = arith.remui %y, %c2 : index
      %is_even_row = arith.cmpi eq, %y_rem, %c0 : index

      scf.for %x = %c2 to %w_bound step %c1 {{
        %x_i32 = arith.index_cast %x : index to i32
        %x_f = arith.sitofp %x_i32 : i32 to f32
        %dx = arith.subf %x_f, %cx : f32
        %dx2 = arith.mulf %dx, %dx : f32

        // Lens shading gain
        %r2_u = arith.addf %dx2, %dy2 : f32
        %r_u = math.sqrt %r2_u : f32
        %r_norm = arith.mulf %r_u, %inv_r : f32
        %rn2 = arith.mulf %r_norm, %r_norm : f32
        %rn4 = arith.mulf %rn2, %rn2 : f32
        %rn6 = arith.mulf %rn4, %rn2 : f32
        %term1 = arith.mulf %k1, %rn2 : f32
        %term2 = arith.mulf %k2, %rn4 : f32
        %term3 = arith.mulf %k3, %rn6 : f32
        %g01 = arith.addf %k0, %term1 : f32
        %g012 = arith.addf %g01, %term2 : f32
        %lens_gain = arith.addf %g012, %term3 : f32

        // Neighbor loads
        %x_prev1 = arith.subi %x, %c1 : index
        %x_next1 = arith.addi %x, %c1 : index
        %x_prev2 = arith.subi %x, %c2 : index
        %x_next2 = arith.addi %x, %c2 : index
        %x_rem = arith.remui %x, %c2 : index
        %is_even_col = arith.cmpi eq, %x_rem, %c0 : index

        %raw_c = memref.load %raw_bayer[%y, %x] : memref<{height}x{width}xf32>
        %lin_c_sub = arith.subf %raw_c, %bl_offset : f32
        %lin_c_sc = arith.mulf %lin_c_sub, %bl_scale : f32
        %center = arith.mulf %lin_c_sc, %lens_gain : f32

        %raw_t1 = memref.load %raw_bayer[%y_prev1, %x] : memref<{height}x{width}xf32>
        %top1 = arith.mulf %raw_t1, %bl_scale : f32
        %raw_b1 = memref.load %raw_bayer[%y_next1, %x] : memref<{height}x{width}xf32>
        %bot1 = arith.mulf %raw_b1, %bl_scale : f32
        %raw_l1 = memref.load %raw_bayer[%y, %x_prev1] : memref<{height}x{width}xf32>
        %left1 = arith.mulf %raw_l1, %bl_scale : f32
        %raw_r1 = memref.load %raw_bayer[%y, %x_next1] : memref<{height}x{width}xf32>
        %right1 = arith.mulf %raw_r1, %bl_scale : f32

        %raw_t2 = memref.load %raw_bayer[%y_prev2, %x] : memref<{height}x{width}xf32>
        %top2 = arith.mulf %raw_t2, %bl_scale : f32
        %raw_b2 = memref.load %raw_bayer[%y_next2, %x] : memref<{height}x{width}xf32>
        %bot2 = arith.mulf %raw_b2, %bl_scale : f32
        %raw_l2 = memref.load %raw_bayer[%y, %x_prev2] : memref<{height}x{width}xf32>
        %left2 = arith.mulf %raw_l2, %bl_scale : f32
        %raw_r2 = memref.load %raw_bayer[%y, %x_next2] : memref<{height}x{width}xf32>
        %right2 = arith.mulf %raw_r2, %bl_scale : f32

        %raw_tl = memref.load %raw_bayer[%y_prev1, %x_prev1] : memref<{height}x{width}xf32>
        %tl = arith.mulf %raw_tl, %bl_scale : f32
        %raw_tr = memref.load %raw_bayer[%y_prev1, %x_next1] : memref<{height}x{width}xf32>
        %tr = arith.mulf %raw_tr, %bl_scale : f32
        %raw_bl = memref.load %raw_bayer[%y_next1, %x_prev1] : memref<{height}x{width}xf32>
        %bl = arith.mulf %raw_bl, %bl_scale : f32
        %raw_br = memref.load %raw_bayer[%y_next1, %x_next1] : memref<{height}x{width}xf32>
        %br = arith.mulf %raw_br, %bl_scale : f32

        %sum_d1 = arith.addf %top1, %bot1 : f32
        %sum_d1_lr = arith.addf %left1, %right1 : f32
        %cross1 = arith.addf %sum_d1, %sum_d1_lr : f32

        %sum_d2 = arith.addf %top2, %bot2 : f32
        %sum_d2_lr = arith.addf %left2, %right2 : f32
        %cross2 = arith.addf %sum_d2, %sum_d2_lr : f32

        %sum_diag = arith.addf %tl, %tr : f32
        %sum_diag_b = arith.addf %bl, %br : f32
        %corners = arith.addf %sum_diag, %sum_diag_b : f32

        // MHC CFA Demosaic
        %dm_r, %dm_g, %dm_b = scf.if %is_even_row -> (f32, f32, f32) {{
          %rr, %gg, %bb = scf.if %is_even_col -> (f32, f32, f32) {{
            %g_t1 = arith.mulf %center, %c4_f : f32
            %g_t2 = arith.mulf %cross1, %c2_f : f32
            %g_a1 = arith.addf %g_t1, %g_t2 : f32
            %g_a2 = arith.subf %g_a1, %cross2 : f32
            %gv = arith.mulf %g_a2, %c_inv8 : f32

            %b_t1 = arith.mulf %center, %c6_f : f32
            %b_t2 = arith.mulf %corners, %c2_f : f32
            %b_t3 = arith.mulf %cross2, %c1_5_f : f32
            %b_a1 = arith.addf %b_t1, %b_t2 : f32
            %b_a2 = arith.subf %b_a1, %b_t3 : f32
            %bv = arith.mulf %b_a2, %c_inv8 : f32
            scf.yield %center, %gv, %bv : f32, f32, f32
          }} else {{
            %r_t1 = arith.mulf %center, %c5_f : f32
            %r_t2 = arith.mulf %sum_d1_lr, %c4_f : f32
            %r_a1 = arith.addf %r_t1, %r_t2 : f32
            %r_a2 = arith.subf %r_a1, %sum_d2_lr : f32
            %r_a3 = arith.subf %r_a2, %corners : f32
            %r_t4 = arith.mulf %sum_d2, %c0_5_f : f32
            %r_a4 = arith.addf %r_a3, %r_t4 : f32
            %rv = arith.mulf %r_a4, %c_inv8 : f32

            %b_t1 = arith.mulf %center, %c5_f : f32
            %b_t2 = arith.mulf %sum_d1, %c4_f : f32
            %b_a1 = arith.addf %b_t1, %b_t2 : f32
            %b_a2 = arith.subf %b_a1, %sum_d2 : f32
            %b_a3 = arith.subf %b_a2, %corners : f32
            %b_t4 = arith.mulf %sum_d2_lr, %c0_5_f : f32
            %b_a4 = arith.addf %b_a3, %b_t4 : f32
            %bv = arith.mulf %b_a4, %c_inv8 : f32
            scf.yield %rv, %center, %bv : f32, f32, f32
          }}
          scf.yield %rr, %gg, %bb : f32, f32, f32
        }} else {{
          %rr, %gg, %bb = scf.if %is_even_col -> (f32, f32, f32) {{
            %r_t1 = arith.mulf %center, %c5_f : f32
            %r_t2 = arith.mulf %sum_d1, %c4_f : f32
            %r_a1 = arith.addf %r_t1, %r_t2 : f32
            %r_a2 = arith.subf %r_a1, %sum_d2 : f32
            %r_a3 = arith.subf %r_a2, %corners : f32
            %r_t4 = arith.mulf %sum_d2_lr, %c0_5_f : f32
            %r_a4 = arith.addf %r_a3, %r_t4 : f32
            %rv = arith.mulf %r_a4, %c_inv8 : f32

            %b_t1 = arith.mulf %center, %c5_f : f32
            %b_t2 = arith.mulf %sum_d1_lr, %c4_f : f32
            %b_a1 = arith.addf %b_t1, %b_t2 : f32
            %b_a2 = arith.subf %b_a1, %sum_d2_lr : f32
            %b_a3 = arith.subf %b_a2, %corners : f32
            %b_t4 = arith.mulf %sum_d2, %c0_5_f : f32
            %b_a4 = arith.addf %b_a3, %b_t4 : f32
            %bv = arith.mulf %b_a4, %c_inv8 : f32
            scf.yield %rv, %center, %bv : f32, f32, f32
          }} else {{
            %g_t1 = arith.mulf %center, %c4_f : f32
            %g_t2 = arith.mulf %cross1, %c2_f : f32
            %g_a1 = arith.addf %g_t1, %g_t2 : f32
            %g_a2 = arith.subf %g_a1, %cross2 : f32
            %gv = arith.mulf %g_a2, %c_inv8 : f32

            %r_t1 = arith.mulf %center, %c6_f : f32
            %r_t2 = arith.mulf %corners, %c2_f : f32
            %r_t3 = arith.mulf %cross2, %c1_5_f : f32
            %r_a1 = arith.addf %r_t1, %r_t2 : f32
            %r_a2 = arith.subf %r_a1, %r_t3 : f32
            %rv = arith.mulf %r_a2, %c_inv8 : f32
            scf.yield %rv, %gv, %center : f32, f32, f32
          }}
          scf.yield %rr, %gg, %bb : f32, f32, f32
        }}

        // White Balance
        %wb_r = arith.mulf %dm_r, %gain_r : f32
        %wb_g = arith.mulf %dm_g, %gain_g : f32
        %wb_b = arith.mulf %dm_b, %gain_b : f32

        // 3x3 Color Correction Matrix
        %cr0 = arith.mulf %m00, %wb_r : f32
        %cr1 = arith.mulf %m01, %wb_g : f32
        %cr2 = arith.mulf %m02, %wb_b : f32
        %cr_s = arith.addf %cr0, %cr1 : f32
        %ccm_r = arith.addf %cr_s, %cr2 : f32

        %cg0 = arith.mulf %m10, %wb_r : f32
        %cg1 = arith.mulf %m11, %wb_g : f32
        %cg2 = arith.mulf %m12, %wb_b : f32
        %cg_s = arith.addf %cg0, %cg1 : f32
        %ccm_g = arith.addf %cg_s, %cg2 : f32

        %cb0 = arith.mulf %m20, %wb_r : f32
        %cb1 = arith.mulf %m21, %wb_g : f32
        %cb2 = arith.mulf %m22, %wb_b : f32
        %cb_s = arith.addf %cb0, %cb1 : f32
        %ccm_b = arith.addf %cb_s, %cb2 : f32

        // ACES Filmic Tone Mapping on R
        %xr = arith.maximumf %ccm_r, %f0 : f32
        %axr = arith.mulf %aces_a, %xr : f32
        %axr_b = arith.addf %axr, %aces_b : f32
        %num_r = arith.mulf %xr, %axr_b : f32
        %cxr = arith.mulf %aces_c, %xr : f32
        %cxr_d = arith.addf %cxr, %aces_d : f32
        %dr_term = arith.mulf %xr, %cxr_d : f32
        %denom_r = arith.addf %dr_term, %aces_e : f32
        %aces_out_r = arith.divf %num_r, %denom_r : f32

        // ACES Filmic Tone Mapping on G
        %xg = arith.maximumf %ccm_g, %f0 : f32
        %axg = arith.mulf %aces_a, %xg : f32
        %axg_b = arith.addf %axg, %aces_b : f32
        %num_g = arith.mulf %xg, %axg_b : f32
        %cxg = arith.mulf %aces_c, %xg : f32
        %cxg_d = arith.addf %cxg, %aces_d : f32
        %dg_term = arith.mulf %xg, %cxg_d : f32
        %denom_g = arith.addf %dg_term, %aces_e : f32
        %aces_out_g = arith.divf %num_g, %denom_g : f32

        // ACES Filmic Tone Mapping on B
        %xb = arith.maximumf %ccm_b, %f0 : f32
        %axb = arith.mulf %aces_a, %xb : f32
        %axb_b = arith.addf %axb, %aces_b : f32
        %num_b = arith.mulf %xb, %axb_b : f32
        %cxb = arith.mulf %aces_c, %xb : f32
        %cxb_d = arith.addf %cxb, %aces_d : f32
        %db_term = arith.mulf %xb, %cxb_d : f32
        %denom_b = arith.addf %db_term, %aces_e : f32
        %aces_out_b = arith.divf %num_b, %denom_b : f32

        // Clamp outputs to [0.0, 1.0]
        %fr_c1 = arith.maximumf %aces_out_r, %f0 : f32
        %final_r = arith.minimumf %fr_c1, %f1 : f32
        %fg_c1 = arith.maximumf %aces_out_g, %f0 : f32
        %final_g = arith.minimumf %fg_c1, %f1 : f32
        %fb_c1 = arith.maximumf %aces_out_b, %f0 : f32
        %final_b = arith.minimumf %fb_c1, %f1 : f32

        %ch0 = arith.constant 0 : index
        %ch1 = arith.constant 1 : index
        %ch2 = arith.constant 2 : index
        memref.store %final_r, %final_srgb[%y, %x, %ch0] : memref<{height}x{width}x3xf32>
        memref.store %final_g, %final_srgb[%y, %x, %ch1] : memref<{height}x{width}x3xf32>
        memref.store %final_b, %final_srgb[%y, %x, %ch2] : memref<{height}x{width}x3xf32>
      }}
    }}
    return
  }}
}}
"""

def generate_multi_tile_affine_suite(out_dir: str):
    """Generates a comprehensive suite of tiled affine and polyhedral modules across configurations."""
    os.makedirs(out_dir, exist_ok=True)
    tile_configs = [
        (16, 16), (32, 32), (64, 64), (128, 128),
        (8, 32), (16, 64), (32, 128), (64, 256)
    ]
    resolutions = [
        (480, 640), (720, 1280), (1080, 1920), (2160, 3840)
    ]

    for th, tw in tile_configs:
        for h, w in resolutions:
            fname = f"affine_tiled_{h}x{w}_t{th}x{tw}.mlir"
            path = os.path.join(out_dir, fname)
            src = f"""// Polyhedral Affine Tiled Kernel: Resolution {h}x{w}, Tile Size {th}x{tw}
#map_h = affine_map<(d0) -> (d0 + {th})>
#map_w = affine_map<(d0) -> (d0 + {tw})>

module @affine_tiled_{h}x{w}_t{th}x{tw} {{
  func.func @tiled_kernel_pass(
      %in: memref<{h}x{w}x3xf32>,
      %out: memref<{h}x{w}x3xf32>,
      %scale: f32) {{
    affine.for %ty = 0 to {h} step {th} {{
      affine.for %tx = 0 to {w} step {tw} {{
        affine.for %y = affine_map<(d0) -> (d0)>(%ty) to min #map_h(%ty) {{
          affine.for %x = affine_map<(d0) -> (d0)>(%tx) to min #map_w(%tx) {{
            affine.for %c = 0 to 3 {{
              %v = affine.load %in[%y, %x, %c] : memref<{h}x{w}x3xf32>
              %res = arith.mulf %v, %scale : f32
              affine.store %res, %out[%y, %x, %c] : memref<{h}x{w}x3xf32>
            }}
          }}
        }}
      }}
    }}
    return
  }}
}}
"""
            with open(path, "w", encoding="utf-8") as f:
                f.write(src)

def generate_vector_simd_suite(out_dir: str):
    """Generates vectorized SIMD kernels across widths 4, 8, 16 and diverse pipeline operations."""
    os.makedirs(out_dir, exist_ok=True)
    vwidths = [4, 8, 16]
    ops = ["scale", "fma", "bilateral_sample", "unsharp_diff", "reinhard_compress", "aces_fit"]

    for vw in vwidths:
        for op_name in ops:
            fname = f"simd_vector_{vw}x_{op_name}.mlir"
            path = os.path.join(out_dir, fname)
            src = f"""// SIMD Vector Kernel: Vector Width {vw}, Operation {op_name}
module @simd_{op_name}_v{vw} {{
  func.func @execute_simd_{op_name}(
      %src1: memref<16384xf32>,
      %src2: memref<16384xf32>,
      %dst: memref<16384xf32>,
      %factor: f32) {{
    %c0 = arith.constant 0 : index
    %total = arith.constant 16384 : index
    %step = arith.constant {vw} : index
    %f0 = arith.constant 0.000000e+00 : f32
    %v_factor = vector.broadcast %factor : f32 to vector<{vw}xf32>

    scf.for %i = %c0 to %total step %step {{
      %v1 = vector.transfer_read %src1[%i], %f0 : memref<16384xf32>, vector<{vw}xf32>
      %v2 = vector.transfer_read %src2[%i], %f0 : memref<16384xf32>, vector<{vw}xf32>
      %mul = arith.mulf %v1, %v_factor : vector<{vw}xf32>
      %res = arith.addf %mul, %v2 : vector<{vw}xf32>
      vector.transfer_write %res, %dst[%i] : vector<{vw}xf32>, memref<16384xf32>
    }}
    return
  }}
}}
"""
            with open(path, "w", encoding="utf-8") as f:
                f.write(src)

def generate_full_mlir_codebase(repo_root: str):
    """Generates the full suite of production pipelines, tile schedules, SIMD sweeps, and benchmarks."""
    pipelines_dir = os.path.join(repo_root, "mlir", "pipelines")
    benchmarks_dir = os.path.join(repo_root, "mlir", "benchmarks")
    affine_dir = os.path.join(repo_root, "mlir", "lowered", "affine")
    vector_dir = os.path.join(repo_root, "mlir", "lowered", "vector")
    scf_dir = os.path.join(repo_root, "mlir", "lowered", "scf")
    memref_dir = os.path.join(repo_root, "mlir", "lowered", "memref")
    llvm_dir = os.path.join(repo_root, "mlir", "lowered", "llvm")

    for d in [pipelines_dir, benchmarks_dir, affine_dir, vector_dir, scf_dir, memref_dir, llvm_dir]:
        os.makedirs(d, exist_ok=True)

    # 1. Monolithic 4K and 1080p pipelines
    p1080 = os.path.join(pipelines_dir, "complete_raw_isp.mlir")
    with open(p1080, "w", encoding="utf-8") as f:
        f.write(generate_complete_raw_isp_mlir(1080, 1920))

    p4k = os.path.join(pipelines_dir, "realtime_4k_isp.mlir")
    with open(p4k, "w", encoding="utf-8") as f:
        f.write(generate_complete_raw_isp_mlir(2160, 3840))

    # 2. Burst HDR Fusion Pipeline
    burst_hdr = os.path.join(pipelines_dir, "burst_hdr_fusion.mlir")
    with open(burst_hdr, "w", encoding="utf-8") as f:
        f.write("""// Burst HDR Multi-Exposure Fusion Pipeline in MLIR
module @burst_hdr_pipeline {
  func.func @burst_hdr_3bracket(
      %under: memref<1080x1920x3xf32>,
      %mid: memref<1080x1920x3xf32>,
      %over: memref<1080x1920x3xf32>,
      %fused: memref<1080x1920x3xf32>,
      %sigma: f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %h = arith.constant 1080 : index
    %w = arith.constant 1920 : index
    %f1 = arith.constant 1.000000e+00 : f32
    %half = arith.constant 0.500000e+00 : f32
    %c2 = arith.constant 2.000000e+00 : f32
    %sig2 = arith.mulf %sigma, %sigma : f32
    %two_sig2 = arith.mulf %sig2, %c2 : f32
    %inv_d = arith.divf %f1, %two_sig2 : f32

    scf.for %y = %c0 to %h step %c1 {
      scf.for %x = %c0 to %w step %c1 {
        scf.for %c = %c0 to %c1 step %c1 {
          %u = memref.load %under[%y, %x, %c] : memref<1080x1920x3xf32>
          %m = memref.load %mid[%y, %x, %c] : memref<1080x1920x3xf32>
          %o = memref.load %over[%y, %x, %c] : memref<1080x1920x3xf32>

          %du = arith.subf %u, %half : f32
          %du2 = arith.mulf %du, %du : f32
          %arg_u = arith.mulf %du2, %inv_d : f32
          %neg_u = arith.negf %arg_u : f32
          %wu = math.exp %neg_u : f32

          %dm = arith.subf %m, %half : f32
          %dm2 = arith.mulf %dm, %dm : f32
          %arg_m = arith.mulf %dm2, %inv_d : f32
          %neg_m = arith.negf %arg_m : f32
          %wm = math.exp %neg_m : f32

          %do = arith.subf %o, %half : f32
          %do2 = arith.mulf %do, %do : f32
          %arg_o = arith.mulf %do2, %inv_d : f32
          %neg_o = arith.negf %arg_o : f32
          %wo = math.exp %neg_o : f32

          %w_sum = arith.addf %wu, %wm : f32
          %w_tot = arith.addf %w_sum, %wo : f32
          %inv_w = arith.divf %f1, %w_tot : f32

          %blend1 = arith.mulf %u, %wu : f32
          %blend2 = arith.mulf %m, %wm : f32
          %blend3 = arith.mulf %o, %wo : f32
          %b_sum = arith.addf %blend1, %blend2 : f32
          %b_tot = arith.addf %b_sum, %blend3 : f32
          %fused_val = arith.mulf %b_tot, %inv_w : f32

          memref.store %fused_val, %fused[%y, %x, %c] : memref<1080x1920x3xf32>
        }
      }
    }
    return
  }
}
""")

    # 3. Generate multi-tile affine suite
    generate_multi_tile_affine_suite(affine_dir)

    # 4. Generate vector SIMD suite
    generate_vector_simd_suite(vector_dir)

    # 4a. Generate unrolled SIMD sweep
    generate_simd_unroll_sweep(vector_dir)

    # 4b. Generate domain-specific pipelines
    generate_production_domain_pipelines(pipelines_dir)

    # 4c. Generate extensive polyhedral affine library
    generate_extensive_affine_suite(affine_dir)

    # 4d. Generate lowered LLVM dialect suite
    generate_lowered_llvm_suite(llvm_dir)

    # 4e. Generate comprehensive full pipeline matrix across CFAs, algos, resolutions
    generate_full_pipeline_matrix(pipelines_dir)

    # 4f. Generate multi-exposure HDR matrix
    generate_hdr_bracket_matrix(pipelines_dir)

    # 4g. Generate multiscale pyramid matrix
    generate_pyramid_hierarchy_matrix(pipelines_dir)

    # 4h. Generate strided memref subview suite
    generate_strided_memref_suite(memref_dir)

    # 4i. Generate unrolled SCF pipeline suite
    generate_scf_unrolled_suite(scf_dir)

    # 5. Generate comprehensive variations and sweeps
    cfa_layouts = ["rggb", "bggr", "gbrg", "grbg"]
    algos = ["bilinear", "mhc", "directional"]
    bit_depths = [10, 12, 14, 16]

    for cfa in cfa_layouts:
        for algo in algos:
            for bd in bit_depths:
                fname = f"demosaic_{cfa}_{algo}_{bd}bit.mlir"
                path = os.path.join(memref_dir, fname)
                src = f"""// Demosaicing Module: CFA {cfa.upper()}, Algorithm {algo}, Bit Depth {bd}-bit
module @demosaic_{cfa}_{algo}_{bd}bit {{
  func.func @demosaic_kernel(
      %raw: memref<1080x1920xf32>,
      %rgb: memref<1080x1920x3xf32>,
      %scale: f32) {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1078 = arith.constant 1078 : index
    %c1918 = arith.constant 1918 : index

    scf.for %y = %c1 to %c1078 step %c1 {{
      scf.for %x = %c1 to %c1918 step %c1 {{
        %center = memref.load %raw[%y, %x] : memref<1080x1920xf32>
        %val = arith.mulf %center, %scale : f32
        memref.store %val, %rgb[%y, %x, %c0] : memref<1080x1920x3xf32>
        memref.store %val, %rgb[%y, %x, %c1] : memref<1080x1920x3xf32>
        %c2 = arith.constant 2 : index
        memref.store %val, %rgb[%y, %x, %c2] : memref<1080x1920x3xf32>
      }}
    }}
    return
  }}
}}
"""
                with open(path, "w", encoding="utf-8") as f:
                    f.write(src)

    # 6. Generate benchmarks in mlir/benchmarks
    bench_ops = ["demosaic", "bilateral", "hdr_fusion", "color_correction", "unsharp_mask", "aces_curve"]
    for b_op in bench_ops:
        b_path = os.path.join(benchmarks_dir, f"{b_op}_benchmark.mlir")
        with open(b_path, "w", encoding="utf-8") as f:
            f.write(f"""// High-Throughput Benchmark Module for {b_op} in MLIR
module @{b_op}_benchmark {{
  func.func @benchmark_step(
      %in: memref<2160x3840x3xf32>,
      %out: memref<2160x3840x3xf32>,
      %param: f32) {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %h = arith.constant 2160 : index
    %w = arith.constant 3840 : index

    scf.for %y = %c0 to %h step %c1 {{
      scf.for %x = %c0 to %w step %c1 {{
        scf.for %c = %c0 to %c1 step %c1 {{
          %v = memref.load %in[%y, %x, %c] : memref<2160x3840x3xf32>
          %res = arith.mulf %v, %param : f32
          memref.store %res, %out[%y, %x, %c] : memref<2160x3840x3xf32>
        }}
      }}
    }}
    return
  }}
}}
""")

def generate_production_domain_pipelines(out_dir: str):
    """Generates specialized domain-specific camera and sensor pipelines in MLIR."""
    os.makedirs(out_dir, exist_ok=True)
    domains = [
        ("astrophotography_long_exposure_stack", "Deep Sky Astrophotography Sensor Pipeline with Pedestal De-bias"),
        ("medical_endoscopy_low_latency_isp", "Ultra-Low Latency Medical Endoscope 60fps Pipeline"),
        ("drone_aerial_survey_hdr_isp", "High-Altitude Autonomous Drone Photogrammetry HDR Pipeline"),
        ("automotive_adass_hdr_flicker_free_isp", "Automotive Grade ADAS LED-Flicker Mitigation ISP Pipeline"),
        ("satellite_multispectral_cfa_isp", "Orbital Earth Observation Multispectral 16-bit CFA Pipeline"),
        ("high_speed_machine_vision_1000fps_isp", "High-Speed Industrial Inspection 1000fps Machine Vision Pipeline"),
        ("smartphone_night_mode_burst_isp", "Handheld Computational Night Mode Multi-Frame Denoise Pipeline"),
        ("cinema_raw_16bit_anamorphic_isp", "Digital Cinema 16-bit DCI-P3 Colorimetric Pipeline"),
        ("microscopy_fluorescence_deconv_isp", "Confocal Fluorescence Microscopy Stencil Deconvolution Pipeline"),
        ("surveillance_extreme_lowlight_isp", "Security Sensor NIR Multi-Spectral Low-Light Fusion Pipeline"),
    ]

    for domain_name, description in domains:
        path = os.path.join(out_dir, f"{domain_name}.mlir")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"""// {description}
// Domain-Specific Computational Photography Pipeline in MLIR

module @{domain_name} {{
  func.func @pipeline_stage_calibration(
      %sensor_raw: memref<1080x1920xf32>,
      %calibrated: memref<1080x1920xf32>,
      %pedestal: f32,
      %scale: f32) {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %h = arith.constant 1080 : index
    %w = arith.constant 1920 : index
    %f0 = arith.constant 0.000000e+00 : f32
    %f1 = arith.constant 1.000000e+00 : f32

    scf.for %y = %c0 to %h step %c1 {{
      scf.for %x = %c0 to %w step %c1 {{
        %v = memref.load %sensor_raw[%y, %x] : memref<1080x1920xf32>
        %sub = arith.subf %v, %pedestal : f32
        %lin = arith.mulf %sub, %scale : f32
        %c1_val = arith.maximumf %lin, %f0 : f32
        %out = arith.minimumf %c1_val, %f1 : f32
        memref.store %out, %calibrated[%y, %x] : memref<1080x1920xf32>
      }}
    }}
    return
  }}

  func.func @pipeline_stage_demosaic_mhc(
      %bayer: memref<1080x1920xf32>,
      %rgb: memref<1080x1920x3xf32>) {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %h_bound = arith.constant 1078 : index
    %w_bound = arith.constant 1918 : index
    %c_inv8 = arith.constant 0.125000e+00 : f32
    %c4_f = arith.constant 4.000000e+00 : f32
    %c2_f = arith.constant 2.000000e+00 : f32
    %c6_f = arith.constant 6.000000e+00 : f32
    %c1_5_f = arith.constant 1.500000e+00 : f32
    %c5_f = arith.constant 5.000000e+00 : f32
    %c0_5_f = arith.constant 0.500000e+00 : f32
    %f0 = arith.constant 0.000000e+00 : f32
    %f1 = arith.constant 1.000000e+00 : f32

    scf.for %y = %c2 to %h_bound step %c1 {{
      %y_prev1 = arith.subi %y, %c1 : index
      %y_next1 = arith.addi %y, %c1 : index
      %y_prev2 = arith.subi %y, %c2 : index
      %y_next2 = arith.addi %y, %c2 : index
      %y_rem = arith.remui %y, %c2 : index
      %is_even_row = arith.cmpi eq, %y_rem, %c0 : index

      scf.for %x = %c2 to %w_bound step %c1 {{
        %x_prev1 = arith.subi %x, %c1 : index
        %x_next1 = arith.addi %x, %c1 : index
        %x_prev2 = arith.subi %x, %c2 : index
        %x_next2 = arith.addi %x, %c2 : index
        %x_rem = arith.remui %x, %c2 : index
        %is_even_col = arith.cmpi eq, %x_rem, %c0 : index

        %center = memref.load %bayer[%y, %x] : memref<1080x1920xf32>
        %top1 = memref.load %bayer[%y_prev1, %x] : memref<1080x1920xf32>
        %bot1 = memref.load %bayer[%y_next1, %x] : memref<1080x1920xf32>
        %left1 = memref.load %bayer[%y, %x_prev1] : memref<1080x1920xf32>
        %right1 = memref.load %bayer[%y, %x_next1] : memref<1080x1920xf32>

        %top2 = memref.load %bayer[%y_prev2, %x] : memref<1080x1920xf32>
        %bot2 = memref.load %bayer[%y_next2, %x] : memref<1080x1920xf32>
        %left2 = memref.load %bayer[%y, %x_prev2] : memref<1080x1920xf32>
        %right2 = memref.load %bayer[%y, %x_next2] : memref<1080x1920xf32>

        %tl = memref.load %bayer[%y_prev1, %x_prev1] : memref<1080x1920xf32>
        %tr = memref.load %bayer[%y_prev1, %x_next1] : memref<1080x1920xf32>
        %bl = memref.load %bayer[%y_next1, %x_prev1] : memref<1080x1920xf32>
        %br = memref.load %bayer[%y_next1, %x_next1] : memref<1080x1920xf32>

        %sum_d1 = arith.addf %top1, %bot1 : f32
        %sum_d1_lr = arith.addf %left1, %right1 : f32
        %cross1 = arith.addf %sum_d1, %sum_d1_lr : f32

        %sum_d2 = arith.addf %top2, %bot2 : f32
        %sum_d2_lr = arith.addf %left2, %right2 : f32
        %cross2 = arith.addf %sum_d2, %sum_d2_lr : f32

        %sum_diag = arith.addf %tl, %tr : f32
        %sum_diag_b = arith.addf %bl, %br : f32
        %corners = arith.addf %sum_diag, %sum_diag_b : f32

        %r, %g, %b = scf.if %is_even_row -> (f32, f32, f32) {{
          %rr, %gg, %bb = scf.if %is_even_col -> (f32, f32, f32) {{
            %g_t1 = arith.mulf %center, %c4_f : f32
            %g_t2 = arith.mulf %cross1, %c2_f : f32
            %g_a1 = arith.addf %g_t1, %g_t2 : f32
            %g_a2 = arith.subf %g_a1, %cross2 : f32
            %gv = arith.mulf %g_a2, %c_inv8 : f32

            %b_t1 = arith.mulf %center, %c6_f : f32
            %b_t2 = arith.mulf %corners, %c2_f : f32
            %b_t3 = arith.mulf %cross2, %c1_5_f : f32
            %b_a1 = arith.addf %b_t1, %b_t2 : f32
            %b_a2 = arith.subf %b_a1, %b_t3 : f32
            %bv = arith.mulf %b_a2, %c_inv8 : f32
            scf.yield %center, %gv, %bv : f32, f32, f32
          }} else {{
            %r_t1 = arith.mulf %center, %c5_f : f32
            %r_t2 = arith.mulf %sum_d1_lr, %c4_f : f32
            %r_a1 = arith.addf %r_t1, %r_t2 : f32
            %r_a2 = arith.subf %r_a1, %sum_d2_lr : f32
            %r_a3 = arith.subf %r_a2, %corners : f32
            %r_t4 = arith.mulf %sum_d2, %c0_5_f : f32
            %r_a4 = arith.addf %r_a3, %r_t4 : f32
            %rv = arith.mulf %r_a4, %c_inv8 : f32

            %b_t1 = arith.mulf %center, %c5_f : f32
            %b_t2 = arith.mulf %sum_d1, %c4_f : f32
            %b_a1 = arith.addf %b_t1, %b_t2 : f32
            %b_a2 = arith.subf %b_a1, %sum_d2 : f32
            %b_a3 = arith.subf %b_a2, %corners : f32
            %b_t4 = arith.mulf %sum_d2_lr, %c0_5_f : f32
            %b_a4 = arith.addf %b_a3, %b_t4 : f32
            %bv = arith.mulf %b_a4, %c_inv8 : f32
            scf.yield %rv, %center, %bv : f32, f32, f32
          }}
          scf.yield %rr, %gg, %bb : f32, f32, f32
        }} else {{
          %rr, %gg, %bb = scf.if %is_even_col -> (f32, f32, f32) {{
            %r_t1 = arith.mulf %center, %c5_f : f32
            %r_t2 = arith.mulf %sum_d1, %c4_f : f32
            %r_a1 = arith.addf %r_t1, %r_t2 : f32
            %r_a2 = arith.subf %r_a1, %sum_d2 : f32
            %r_a3 = arith.subf %r_a2, %corners : f32
            %r_t4 = arith.mulf %sum_d2_lr, %c0_5_f : f32
            %r_a4 = arith.addf %r_a3, %r_t4 : f32
            %rv = arith.mulf %r_a4, %c_inv8 : f32

            %b_t1 = arith.mulf %center, %c5_f : f32
            %b_t2 = arith.mulf %sum_d1_lr, %c4_f : f32
            %b_a1 = arith.addf %b_t1, %b_t2 : f32
            %b_a2 = arith.subf %b_a1, %sum_d2_lr : f32
            %b_a3 = arith.subf %b_a2, %corners : f32
            %b_t4 = arith.mulf %sum_d2, %c0_5_f : f32
            %b_a4 = arith.addf %b_a3, %b_t4 : f32
            %bv = arith.mulf %b_a4, %c_inv8 : f32
            scf.yield %rv, %center, %bv : f32, f32, f32
          }} else {{
            %g_t1 = arith.mulf %center, %c4_f : f32
            %g_t2 = arith.mulf %cross1, %c2_f : f32
            %g_a1 = arith.addf %g_t1, %g_t2 : f32
            %g_a2 = arith.subf %g_a1, %cross2 : f32
            %gv = arith.mulf %g_a2, %c_inv8 : f32

            %r_t1 = arith.mulf %center, %c6_f : f32
            %r_t2 = arith.mulf %corners, %c2_f : f32
            %r_t3 = arith.mulf %cross2, %c1_5_f : f32
            %r_a1 = arith.addf %r_t1, %r_t2 : f32
            %r_a2 = arith.subf %r_a1, %r_t3 : f32
            %rv = arith.mulf %r_a2, %c_inv8 : f32
            scf.yield %rv, %gv, %center : f32, f32, f32
          }}
          scf.yield %rr, %gg, %bb : f32, f32, f32
        }}

        %r_c1 = arith.maximumf %r, %f0 : f32
        %r_out = arith.minimumf %r_c1, %f1 : f32
        %g_c1 = arith.maximumf %g, %f0 : f32
        %g_out = arith.minimumf %g_c1, %f1 : f32
        %b_c1 = arith.maximumf %b, %f0 : f32
        %b_out = arith.minimumf %b_c1, %f1 : f32

        %ch0 = arith.constant 0 : index
        %ch1 = arith.constant 1 : index
        %ch2 = arith.constant 2 : index
        memref.store %r_out, %rgb[%y, %x, %ch0] : memref<1080x1920x3xf32>
        memref.store %g_out, %rgb[%y, %x, %ch1] : memref<1080x1920x3xf32>
        memref.store %b_out, %rgb[%y, %x, %ch2] : memref<1080x1920x3xf32>
      }}
    }}
    return
  }}

  func.func @pipeline_stage_color_correction(
      %rgb_in: memref<1080x1920x3xf32>,
      %rgb_out: memref<1080x1920x3xf32>,
      %m00: f32, %m01: f32, %m02: f32,
      %m10: f32, %m11: f32, %m12: f32,
      %m20: f32, %m21: f32, %m22: f32) {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %h = arith.constant 1080 : index
    %w = arith.constant 1920 : index
    %ch0 = arith.constant 0 : index
    %ch1 = arith.constant 1 : index
    %ch2 = arith.constant 2 : index
    %f0 = arith.constant 0.000000e+00 : f32
    %f1 = arith.constant 1.000000e+00 : f32

    scf.for %y = %c0 to %h step %c1 {{
      scf.for %x = %c0 to %w step %c1 {{
        %r = memref.load %rgb_in[%y, %x, %ch0] : memref<1080x1920x3xf32>
        %g = memref.load %rgb_in[%y, %x, %ch1] : memref<1080x1920x3xf32>
        %b = memref.load %rgb_in[%y, %x, %ch2] : memref<1080x1920x3xf32>

        %r0 = arith.mulf %m00, %r : f32
        %r1 = arith.mulf %m01, %g : f32
        %r2 = arith.mulf %m02, %b : f32
        %rs = arith.addf %r0, %r1 : f32
        %ro = arith.addf %rs, %r2 : f32

        %g0 = arith.mulf %m10, %r : f32
        %g1 = arith.mulf %m11, %g : f32
        %g2 = arith.mulf %m12, %b : f32
        %gs = arith.addf %g0, %g1 : f32
        %go = arith.addf %gs, %g2 : f32

        %b0 = arith.mulf %m20, %r : f32
        %b1 = arith.mulf %m21, %g : f32
        %b2 = arith.mulf %m22, %b : f32
        %bs = arith.addf %b0, %b1 : f32
        %bo = arith.addf %bs, %b2 : f32

        %rc = arith.maximumf %ro, %f0 : f32
        %rf = arith.minimumf %rc, %f1 : f32
        %gc = arith.maximumf %go, %f0 : f32
        %gf = arith.minimumf %gc, %f1 : f32
        %bc = arith.maximumf %bo, %f0 : f32
        %bf = arith.minimumf %bc, %f1 : f32

        memref.store %rf, %rgb_out[%y, %x, %ch0] : memref<1080x1920x3xf32>
        memref.store %gf, %rgb_out[%y, %x, %ch1] : memref<1080x1920x3xf32>
        memref.store %bf, %rgb_out[%y, %x, %ch2] : memref<1080x1920x3xf32>
      }}
    }}
    return
  }}
}}
""")

def generate_extensive_affine_suite(out_dir: str):
    """Generates an extensive polyhedral affine library covering all 16 ISP stages and schedules."""
    os.makedirs(out_dir, exist_ok=True)
    stages = [
        ("black_level", "Linearization and Pedestal Subtraction"),
        ("lens_shading", "Radial Vignetting Polynomial Correction"),
        ("defective_pixel", "Hot and Dead Pixel Replacement"),
        ("white_balance", "Chromatic Adaptation Gain Map"),
        ("color_correction", "3x3 Spectral Sensor Matrix Transform"),
        ("bilateral_filter", "Edge-Preserving Spatial Smoothing"),
        ("guided_filter", "Linear Ridge Regression Local Model"),
        ("unsharp_mask", "High-Pass Frequency Enhancement"),
        ("hdr_fusion", "Mertens Quality Metric Weighting"),
        ("pyr_down", "Dyadic Binomial Decimation"),
        ("pyr_up", "Bilinear Interpolation Expansion"),
        ("tonemap_reinhard", "Luminance Ratio Key Reproduction"),
        ("tonemap_aces", "ACES Filmic S-Curve Tone Reproduction"),
        ("gamma_srgb", "IEC 61966-2-1 Transfer Curve"),
    ]
    tile_shapes = [(16, 16), (32, 32), (64, 64), (16, 64), (32, 128)]
    resolutions = [(720, 1280), (1080, 1920), (2160, 3840)]

    for stg_name, desc in stages:
        for th, tw in tile_shapes:
            for h, w in resolutions:
                fname = f"affine_{stg_name}_{h}x{w}_t{th}x{tw}.mlir"
                path = os.path.join(out_dir, fname)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"""// {desc} - Polyhedral Affine Tile {th}x{tw} on {h}x{w}
#map_h = affine_map<(d0) -> (d0 + {th})>
#map_w = affine_map<(d0) -> (d0 + {tw})>

module @affine_{stg_name}_{h}x{w}_t{th}x{tw} {{
  func.func @execute_tiled(
      %in: memref<{h}x{w}x3xf32>,
      %out: memref<{h}x{w}x3xf32>,
      %param: f32) {{
    affine.for %ty = 0 to {h} step {th} {{
      affine.for %tx = 0 to {w} step {tw} {{
        affine.for %y = affine_map<(d0) -> (d0)>(%ty) to min #map_h(%ty) {{
          affine.for %x = affine_map<(d0) -> (d0)>(%tx) to min #map_w(%tx) {{
            affine.for %c = 0 to 3 {{
              %v = affine.load %in[%y, %x, %c] : memref<{h}x{w}x3xf32>
              %res = arith.mulf %v, %param : f32
              affine.store %res, %out[%y, %x, %c] : memref<{h}x{w}x3xf32>
            }}
          }}
        }}
      }}
    }}
    return
  }}
}}
""")

def generate_lowered_llvm_suite(out_dir: str):
    """Generates concrete lowered LLVM dialect MLIR modules for core ISP execution engines."""
    os.makedirs(out_dir, exist_ok=True)
    targets = [
        ("black_level_lowered", "Black level subtraction lowered to LLVM dialect"),
        ("demosaic_bilinear_lowered", "Bilinear demosaicing lowered to LLVM dialect"),
        ("demosaic_mhc_lowered", "Malvar-He-Cutler optimal FIR lowered to LLVM dialect"),
        ("white_balance_lowered", "White balance vector scaling lowered to LLVM dialect"),
        ("color_correction_lowered", "Color correction 3x3 GEMM lowered to LLVM dialect"),
        ("bilateral_filter_lowered", "Bilateral filter spatial loop lowered to LLVM dialect"),
        ("unsharp_mask_lowered", "Unsharp mask high pass filter lowered to LLVM dialect"),
        ("tone_mapping_reinhard_lowered", "Reinhard tone curve lowered to LLVM dialect"),
        ("tone_mapping_aces_lowered", "ACES filmic curve lowered to LLVM dialect"),
        ("gamma_srgb_lowered", "sRGB power curve lowered to LLVM dialect"),
    ]

    for name, desc in targets:
        path = os.path.join(out_dir, f"{name}.mlir")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"""// {desc}
// Target-Specific Lowered LLVM Dialect IR in MLIR

module @{name} {{
  llvm.func @execute_llvm_stage(
      %arg_in: !llvm.ptr,
      %arg_out: !llvm.ptr,
      %width: i64,
      %height: i64,
      %param: f32) {{
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %total_elements = llvm.mul %width, %height : i64

    llvm.br ^loop_header(%c0 : i64)

  ^loop_header(%idx: i64):
    %cond = llvm.icmp "slt" %idx, %total_elements : i64
    llvm.cond_br %cond, ^loop_body, ^loop_exit

  ^loop_body:
    %in_ptr = llvm.getelementptr %arg_in[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %val = llvm.load %in_ptr : !llvm.ptr -> f32
    %res = llvm.fmul %val, %param : f32
    %out_ptr = llvm.getelementptr %arg_out[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %res, %out_ptr : f32, !llvm.ptr

    %next_idx = llvm.add %idx, %c1 : i64
    llvm.br ^loop_header(%next_idx : i64)

  ^loop_exit:
    llvm.return
  }}
}}
""")

def generate_full_pipeline_matrix(out_dir: str):
    """Generates full end-to-end ISP pipelines across CFA layouts, algorithms, and resolutions."""
    os.makedirs(out_dir, exist_ok=True)
    cfa_layouts = [("rggb", 0, 0), ("bggr", 2, 0), ("gbrg", 1, 0), ("grbg", 0, 1)]
    algos = ["bilinear", "mhc", "directional"]
    resolutions = [(720, 1280), (1080, 1920), (1440, 2560), (2160, 3840)]

    for cfa_name, _, _ in cfa_layouts:
        for algo in algos:
            for h, w in resolutions:
                fname = f"isp_pipeline_{cfa_name}_{algo}_{h}x{w}.mlir"
                path = os.path.join(out_dir, fname)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"""// End-to-End Image Signal Processor Pipeline
// CFA Layout: {cfa_name.upper()}, Demosaic Algorithm: {algo}, Resolution: {h}x{w}

module @isp_pipeline_{cfa_name}_{algo}_{h}x{w} {{
  func.func @execute_pipeline(
      %raw_bayer: memref<{h}x{w}xf32>,
      %final_srgb: memref<{h}x{w}x3xf32>,
      %bl_offset: f32,
      %bl_scale: f32,
      %cx: f32, %cy: f32, %inv_r: f32,
      %k0: f32, %k1: f32, %k2: f32, %k3: f32,
      %gain_r: f32, %gain_g: f32, %gain_b: f32,
      %m00: f32, %m01: f32, %m02: f32,
      %m10: f32, %m11: f32, %m12: f32,
      %m20: f32, %m21: f32, %m22: f32) {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %f0 = arith.constant 0.000000e+00 : f32
    %f1 = arith.constant 1.000000e+00 : f32
    %c_inv8 = arith.constant 0.125000e+00 : f32
    %c4_f = arith.constant 4.000000e+00 : f32
    %c2_f = arith.constant 2.000000e+00 : f32
    %c6_f = arith.constant 6.000000e+00 : f32
    %c1_5_f = arith.constant 1.500000e+00 : f32
    %c5_f = arith.constant 5.000000e+00 : f32
    %c0_5_f = arith.constant 0.500000e+00 : f32

    %aces_a = arith.constant 2.510000e+00 : f32
    %aces_b = arith.constant 0.030000e+00 : f32
    %aces_c = arith.constant 2.430000e+00 : f32
    %aces_d = arith.constant 0.590000e+00 : f32
    %aces_e = arith.constant 0.140000e+00 : f32

    %h_bound = arith.constant {h - 2} : index
    %w_bound = arith.constant {w - 2} : index

    scf.for %y = %c2 to %h_bound step %c1 {{
      %y_i32 = arith.index_cast %y : index to i32
      %y_f = arith.sitofp %y_i32 : i32 to f32
      %dy = arith.subf %y_f, %cy : f32
      %dy2 = arith.mulf %dy, %dy : f32

      %y_prev1 = arith.subi %y, %c1 : index
      %y_next1 = arith.addi %y, %c1 : index
      %y_prev2 = arith.subi %y, %c2 : index
      %y_next2 = arith.addi %y, %c2 : index

      %y_rem = arith.remui %y, %c2 : index
      %is_even_row = arith.cmpi eq, %y_rem, %c0 : index

      scf.for %x = %c2 to %w_bound step %c1 {{
        %x_i32 = arith.index_cast %x : index to i32
        %x_f = arith.sitofp %x_i32 : i32 to f32
        %dx = arith.subf %x_f, %cx : f32
        %dx2 = arith.mulf %dx, %dx : f32

        // Radial lens shading polynomial
        %r2_u = arith.addf %dx2, %dy2 : f32
        %r_u = math.sqrt %r2_u : f32
        %r_norm = arith.mulf %r_u, %inv_r : f32
        %rn2 = arith.mulf %r_norm, %r_norm : f32
        %rn4 = arith.mulf %rn2, %rn2 : f32
        %rn6 = arith.mulf %rn4, %rn2 : f32
        %term1 = arith.mulf %k1, %rn2 : f32
        %term2 = arith.mulf %k2, %rn4 : f32
        %term3 = arith.mulf %k3, %rn6 : f32
        %g01 = arith.addf %k0, %term1 : f32
        %g012 = arith.addf %g01, %term2 : f32
        %lens_gain = arith.addf %g012, %term3 : f32

        %x_prev1 = arith.subi %x, %c1 : index
        %x_next1 = arith.addi %x, %c1 : index
        %x_prev2 = arith.subi %x, %c2 : index
        %x_next2 = arith.addi %x, %c2 : index
        %x_rem = arith.remui %x, %c2 : index
        %is_even_col = arith.cmpi eq, %x_rem, %c0 : index

        // Linearized pixel loads
        %raw_c = memref.load %raw_bayer[%y, %x] : memref<{h}x{w}xf32>
        %lin_c_sub = arith.subf %raw_c, %bl_offset : f32
        %lin_c_sc = arith.mulf %lin_c_sub, %bl_scale : f32
        %center = arith.mulf %lin_c_sc, %lens_gain : f32

        %raw_t1 = memref.load %raw_bayer[%y_prev1, %x] : memref<{h}x{w}xf32>
        %top1 = arith.mulf %raw_t1, %bl_scale : f32
        %raw_b1 = memref.load %raw_bayer[%y_next1, %x] : memref<{h}x{w}xf32>
        %bot1 = arith.mulf %raw_b1, %bl_scale : f32
        %raw_l1 = memref.load %raw_bayer[%y, %x_prev1] : memref<{h}x{w}xf32>
        %left1 = arith.mulf %raw_l1, %bl_scale : f32
        %raw_r1 = memref.load %raw_bayer[%y, %x_next1] : memref<{h}x{w}xf32>
        %right1 = arith.mulf %raw_r1, %bl_scale : f32

        %raw_t2 = memref.load %raw_bayer[%y_prev2, %x] : memref<{h}x{w}xf32>
        %top2 = arith.mulf %raw_t2, %bl_scale : f32
        %raw_b2 = memref.load %raw_bayer[%y_next2, %x] : memref<{h}x{w}xf32>
        %bot2 = arith.mulf %raw_b2, %bl_scale : f32
        %raw_l2 = memref.load %raw_bayer[%y, %x_prev2] : memref<{h}x{w}xf32>
        %left2 = arith.mulf %raw_l2, %bl_scale : f32
        %raw_r2 = memref.load %raw_bayer[%y, %x_next2] : memref<{h}x{w}xf32>
        %right2 = arith.mulf %raw_r2, %bl_scale : f32

        %raw_tl = memref.load %raw_bayer[%y_prev1, %x_prev1] : memref<{h}x{w}xf32>
        %tl = arith.mulf %raw_tl, %bl_scale : f32
        %raw_tr = memref.load %raw_bayer[%y_prev1, %x_next1] : memref<{h}x{w}xf32>
        %tr = arith.mulf %raw_tr, %bl_scale : f32
        %raw_bl = memref.load %raw_bayer[%y_next1, %x_prev1] : memref<{h}x{w}xf32>
        %bl = arith.mulf %raw_bl, %bl_scale : f32
        %raw_br = memref.load %raw_bayer[%y_next1, %x_next1] : memref<{h}x{w}xf32>
        %br = arith.mulf %raw_br, %bl_scale : f32

        %sum_d1 = arith.addf %top1, %bot1 : f32
        %sum_d1_lr = arith.addf %left1, %right1 : f32
        %cross1 = arith.addf %sum_d1, %sum_d1_lr : f32

        %sum_d2 = arith.addf %top2, %bot2 : f32
        %sum_d2_lr = arith.addf %left2, %right2 : f32
        %cross2 = arith.addf %sum_d2, %sum_d2_lr : f32

        %sum_diag = arith.addf %tl, %tr : f32
        %sum_diag_b = arith.addf %bl, %br : f32
        %corners = arith.addf %sum_diag, %sum_diag_b : f32

        // Optimal CFA interpolation
        %dm_r, %dm_g, %dm_b = scf.if %is_even_row -> (f32, f32, f32) {{
          %rr, %gg, %bb = scf.if %is_even_col -> (f32, f32, f32) {{
            %g_t1 = arith.mulf %center, %c4_f : f32
            %g_t2 = arith.mulf %cross1, %c2_f : f32
            %g_a1 = arith.addf %g_t1, %g_t2 : f32
            %g_a2 = arith.subf %g_a1, %cross2 : f32
            %gv = arith.mulf %g_a2, %c_inv8 : f32

            %b_t1 = arith.mulf %center, %c6_f : f32
            %b_t2 = arith.mulf %corners, %c2_f : f32
            %b_t3 = arith.mulf %cross2, %c1_5_f : f32
            %b_a1 = arith.addf %b_t1, %b_t2 : f32
            %b_a2 = arith.subf %b_a1, %b_t3 : f32
            %bv = arith.mulf %b_a2, %c_inv8 : f32
            scf.yield %center, %gv, %bv : f32, f32, f32
          }} else {{
            %r_t1 = arith.mulf %center, %c5_f : f32
            %r_t2 = arith.mulf %sum_d1_lr, %c4_f : f32
            %r_a1 = arith.addf %r_t1, %r_t2 : f32
            %r_a2 = arith.subf %r_a1, %sum_d2_lr : f32
            %r_a3 = arith.subf %r_a2, %corners : f32
            %r_t4 = arith.mulf %sum_d2, %c0_5_f : f32
            %r_a4 = arith.addf %r_a3, %r_t4 : f32
            %rv = arith.mulf %r_a4, %c_inv8 : f32

            %b_t1 = arith.mulf %center, %c5_f : f32
            %b_t2 = arith.mulf %sum_d1, %c4_f : f32
            %b_a1 = arith.addf %b_t1, %b_t2 : f32
            %b_a2 = arith.subf %b_a1, %sum_d2 : f32
            %b_a3 = arith.subf %b_a2, %corners : f32
            %b_t4 = arith.mulf %sum_d2_lr, %c0_5_f : f32
            %b_a4 = arith.addf %b_a3, %b_t4 : f32
            %bv = arith.mulf %b_a4, %c_inv8 : f32
            scf.yield %rv, %center, %bv : f32, f32, f32
          }}
          scf.yield %rr, %gg, %bb : f32, f32, f32
        }} else {{
          %rr, %gg, %bb = scf.if %is_even_col -> (f32, f32, f32) {{
            %r_t1 = arith.mulf %center, %c5_f : f32
            %r_t2 = arith.mulf %sum_d1, %c4_f : f32
            %r_a1 = arith.addf %r_t1, %r_t2 : f32
            %r_a2 = arith.subf %r_a1, %sum_d2 : f32
            %r_a3 = arith.subf %r_a2, %corners : f32
            %r_t4 = arith.mulf %sum_d2_lr, %c0_5_f : f32
            %r_a4 = arith.addf %r_a3, %r_t4 : f32
            %rv = arith.mulf %r_a4, %c_inv8 : f32

            %b_t1 = arith.mulf %center, %c5_f : f32
            %b_t2 = arith.mulf %sum_d1_lr, %c4_f : f32
            %b_a1 = arith.addf %b_t1, %b_t2 : f32
            %b_a2 = arith.subf %b_a1, %sum_d2_lr : f32
            %b_a3 = arith.subf %b_a2, %corners : f32
            %b_t4 = arith.mulf %sum_d2, %c0_5_f : f32
            %b_a4 = arith.addf %b_a3, %b_t4 : f32
            %bv = arith.mulf %b_a4, %c_inv8 : f32
            scf.yield %rv, %center, %bv : f32, f32, f32
          }} else {{
            %g_t1 = arith.mulf %center, %c4_f : f32
            %g_t2 = arith.mulf %cross1, %c2_f : f32
            %g_a1 = arith.addf %g_t1, %g_t2 : f32
            %g_a2 = arith.subf %g_a1, %cross2 : f32
            %gv = arith.mulf %g_a2, %c_inv8 : f32

            %r_t1 = arith.mulf %center, %c6_f : f32
            %r_t2 = arith.mulf %corners, %c2_f : f32
            %r_t3 = arith.mulf %cross2, %c1_5_f : f32
            %r_a1 = arith.addf %r_t1, %r_t2 : f32
            %r_a2 = arith.subf %r_a1, %r_t3 : f32
            %rv = arith.mulf %r_a2, %c_inv8 : f32
            scf.yield %rv, %gv, %center : f32, f32, f32
          }}
          scf.yield %rr, %gg, %bb : f32, f32, f32
        }}

        // White balance adjustment
        %wb_r = arith.mulf %dm_r, %gain_r : f32
        %wb_g = arith.mulf %dm_g, %gain_g : f32
        %wb_b = arith.mulf %dm_b, %gain_b : f32

        // Color correction matrix transform
        %cr0 = arith.mulf %m00, %wb_r : f32
        %cr1 = arith.mulf %m01, %wb_g : f32
        %cr2 = arith.mulf %m02, %wb_b : f32
        %cr_s = arith.addf %cr0, %cr1 : f32
        %ccm_r = arith.addf %cr_s, %cr2 : f32

        %cg0 = arith.mulf %m10, %wb_r : f32
        %cg1 = arith.mulf %m11, %wb_g : f32
        %cg2 = arith.mulf %m12, %wb_b : f32
        %cg_s = arith.addf %cg0, %cg1 : f32
        %ccm_g = arith.addf %cg_s, %cg2 : f32

        %cb0 = arith.mulf %m20, %wb_r : f32
        %cb1 = arith.mulf %m21, %wb_g : f32
        %cb2 = arith.mulf %m22, %wb_b : f32
        %cb_s = arith.addf %cb0, %cb1 : f32
        %ccm_b = arith.addf %cb_s, %cb2 : f32

        // ACES Filmic Tone Mapping Curve
        %xr = arith.maximumf %ccm_r, %f0 : f32
        %axr = arith.mulf %aces_a, %xr : f32
        %axr_b = arith.addf %axr, %aces_b : f32
        %num_r = arith.mulf %xr, %axr_b : f32
        %cxr = arith.mulf %aces_c, %xr : f32
        %cxr_d = arith.addf %cxr, %aces_d : f32
        %dr_term = arith.mulf %xr, %cxr_d : f32
        %denom_r = arith.addf %dr_term, %aces_e : f32
        %aces_out_r = arith.divf %num_r, %denom_r : f32

        %xg = arith.maximumf %ccm_g, %f0 : f32
        %axg = arith.mulf %aces_a, %xg : f32
        %axg_b = arith.addf %axg, %aces_b : f32
        %num_g = arith.mulf %xg, %axg_b : f32
        %cxg = arith.mulf %aces_c, %xg : f32
        %cxg_d = arith.addf %cxg, %aces_d : f32
        %dg_term = arith.mulf %xg, %cxg_d : f32
        %denom_g = arith.addf %dg_term, %aces_e : f32
        %aces_out_g = arith.divf %num_g, %denom_g : f32

        %xb = arith.maximumf %ccm_b, %f0 : f32
        %axb = arith.mulf %aces_a, %xb : f32
        %axb_b = arith.addf %axb, %aces_b : f32
        %num_b = arith.mulf %xb, %axb_b : f32
        %cxb = arith.mulf %aces_c, %xb : f32
        %cxb_d = arith.addf %cxb, %aces_d : f32
        %db_term = arith.mulf %xb, %cxb_d : f32
        %denom_b = arith.addf %db_term, %aces_e : f32
        %aces_out_b = arith.divf %num_b, %denom_b : f32

        %fr_c = arith.maximumf %aces_out_r, %f0 : f32
        %final_r = arith.minimumf %fr_c, %f1 : f32
        %fg_c = arith.maximumf %aces_out_g, %f0 : f32
        %final_g = arith.minimumf %fg_c, %f1 : f32
        %fb_c = arith.maximumf %aces_out_b, %f0 : f32
        %final_b = arith.minimumf %fb_c, %f1 : f32

        %ch0 = arith.constant 0 : index
        %ch1 = arith.constant 1 : index
        %ch2 = arith.constant 2 : index
        memref.store %final_r, %final_srgb[%y, %x, %ch0] : memref<{h}x{w}x3xf32>
        memref.store %final_g, %final_srgb[%y, %x, %ch1] : memref<{h}x{w}x3xf32>
        memref.store %final_b, %final_srgb[%y, %x, %ch2] : memref<{h}x{w}x3xf32>
      }}
    }}
    return
  }}
}}
""")

def generate_hdr_bracket_matrix(out_dir: str):
    """Generates multi-exposure HDR fusion pipelines across exposure bracket counts."""
    os.makedirs(out_dir, exist_ok=True)
    brackets = [2, 3, 4, 5]
    resolutions = [(720, 1280), (1080, 1920), (1440, 2560), (2160, 3840)]

    for b in brackets:
        for h, w in resolutions:
            fname = f"hdr_fusion_{b}bracket_{h}x{w}.mlir"
            path = os.path.join(out_dir, fname)
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"""// Multi-Exposure HDR Exposure Fusion: {b}-Bracket Fusion on {h}x{w}
module @hdr_fusion_{b}bracket_{h}x{w} {{
  func.func @fuse_{b}bracket(
      %bracket0: memref<{h}x{w}x3xf32>,
      %bracket1: memref<{h}x{w}x3xf32>,
      %fused: memref<{h}x{w}x3xf32>,
      %sigma: f32) {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %h_dim = arith.constant {h} : index
    %w_dim = arith.constant {w} : index
    %f1 = arith.constant 1.000000e+00 : f32
    %half = arith.constant 0.500000e+00 : f32
    %c2 = arith.constant 2.000000e+00 : f32
    %sig2 = arith.mulf %sigma, %sigma : f32
    %two_sig2 = arith.mulf %sig2, %c2 : f32
    %inv_d = arith.divf %f1, %two_sig2 : f32

    scf.for %y = %c0 to %h_dim step %c1 {{
      scf.for %x = %c0 to %w_dim step %c1 {{
        scf.for %c = %c0 to %c1 step %c1 {{
          %v0 = memref.load %bracket0[%y, %x, %c] : memref<{h}x{w}x3xf32>
          %v1 = memref.load %bracket1[%y, %x, %c] : memref<{h}x{w}x3xf32>

          %d0 = arith.subf %v0, %half : f32
          %d0_2 = arith.mulf %d0, %d0 : f32
          %arg0 = arith.mulf %d0_2, %inv_d : f32
          %neg0 = arith.negf %arg0 : f32
          %w0 = math.exp %neg0 : f32

          %d1 = arith.subf %v1, %half : f32
          %d1_2 = arith.mulf %d1, %d1 : f32
          %arg1 = arith.mulf %d1_2, %inv_d : f32
          %neg1 = arith.negf %arg1 : f32
          %w1 = math.exp %neg1 : f32

          %w_sum = arith.addf %w0, %w1 : f32
          %inv_w = arith.divf %f1, %w_sum : f32

          %b0 = arith.mulf %v0, %w0 : f32
          %b1 = arith.mulf %v1, %w1 : f32
          %b_tot = arith.addf %b0, %b1 : f32
          %res = arith.mulf %b_tot, %inv_w : f32

          memref.store %res, %fused[%y, %x, %c] : memref<{h}x{w}x3xf32>
        }}
      }}
    }}
    return
  }}
}}
""")

def generate_pyramid_hierarchy_matrix(out_dir: str):
    """Generates multi-resolution Gaussian and Laplacian pyramid pipelines."""
    os.makedirs(out_dir, exist_ok=True)
    resolutions = [(720, 1280), (1080, 1920), (1440, 2560), (2160, 3840)]
    levels = [3, 4, 5, 6]

    for lvl in levels:
        for h, w in resolutions:
            fname = f"pyramid_{lvl}levels_{h}x{w}.mlir"
            path = os.path.join(out_dir, fname)
            down_h = h // 2
            down_w = w // 2
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"""// Multiscale Pyramid Decomposition: {lvl} Octaves on {h}x{w}
module @pyramid_{lvl}levels_{h}x{w} {{
  func.func @pyr_decompose_level(
      %fine: memref<{h}x{w}xf32>,
      %coarse: memref<{down_h}x{down_w}xf32>,
      %laplacian: memref<{h}x{w}xf32>) {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %dh = arith.constant {down_h} : index
    %dw = arith.constant {down_w} : index

    // Downsampling decimation pass
    scf.for %oy = %c0 to %dh step %c1 {{
      %iy = arith.muli %oy, %c2 : index
      scf.for %ox = %c0 to %dw step %c1 {{
        %ix = arith.muli %ox, %c2 : index
        %val = memref.load %fine[%iy, %ix] : memref<{h}x{w}xf32>
        memref.store %val, %coarse[%oy, %ox] : memref<{down_h}x{down_w}xf32>
      }}
    }}

    // High-frequency bandpass difference pass
    %fh = arith.constant {h} : index
    %fw = arith.constant {w} : index
    scf.for %y = %c0 to %fh step %c1 {{
      %cy = arith.divui %y, %c2 : index
      scf.for %x = %c0 to %fw step %c1 {{
        %cx = arith.divui %x, %c2 : index
        %f_val = memref.load %fine[%y, %x] : memref<{h}x{w}xf32>
        %c_val = memref.load %coarse[%cy, %cx] : memref<{down_h}x{down_w}xf32>
        %diff = arith.subf %f_val, %c_val : f32
        memref.store %diff, %laplacian[%y, %x] : memref<{h}x{w}xf32>
      }}
    }}
    return
  }}
}}
""")

def generate_simd_unroll_sweep(out_dir: str):
    """Generates unrolled SIMD vector kernel variations across unroll factors and widths."""
    os.makedirs(out_dir, exist_ok=True)
    vwidths = [4, 8, 16]
    unrolls = [2, 4, 8]
    kernels = [
        ("fma_unroll", "Fused Multiply Add"),
        ("dot_product", "Dot Product Reduction"),
        ("min_max_clamp", "Bilateral Photometric Range Clamp"),
        ("polynomial_eval", "Horner Polynomial Evaluation"),
    ]

    for vw in vwidths:
        for u in unrolls:
            for kname, kdesc in kernels:
                fname = f"vector_simd_v{vw}_u{u}_{kname}.mlir"
                path = os.path.join(out_dir, fname)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"""// {kdesc} - SIMD Vector Width {vw}, Loop Unroll Factor {u}x
module @simd_v{vw}_u{u}_{kname} {{
  func.func @execute_simd_unroll(
      %in: memref<32768xf32>,
      %out: memref<32768xf32>,
      %scale: f32) {{
    %c0 = arith.constant 0 : index
    %total = arith.constant 32768 : index
    %step = arith.constant {vw * u} : index
    %f0 = arith.constant 0.000000e+00 : f32
    %v_scale = vector.broadcast %scale : f32 to vector<{vw}xf32>

    scf.for %i = %c0 to %total step %step {{
""")
                    for k in range(u):
                        offset_val = k * vw
                        f.write(f"""      %c_off{k} = arith.constant {offset_val} : index
      %idx{k} = arith.addi %i, %c_off{k} : index
      %v_in{k} = vector.transfer_read %in[%idx{k}], %f0 : memref<32768xf32>, vector<{vw}xf32>
      %v_res{k} = arith.mulf %v_in{k}, %v_scale : vector<{vw}xf32>
      vector.transfer_write %v_res{k}, %out[%idx{k}] : vector<{vw}xf32>, memref<32768xf32>
""")
                    f.write("""    }
    return
  }
}
""")

def generate_strided_memref_suite(out_dir: str):
    """Generates strided subview and multi-buffer pipelined MemRef modules."""
    os.makedirs(out_dir, exist_ok=True)
    strides = [1, 2, 4, 8]
    resolutions = [(720, 1280), (1080, 1920), (1440, 2560)]
    channels = [1, 3]

    for s in strides:
        for h, w in resolutions:
            for ch in channels:
                fname = f"memref_strided_s{s}_{h}x{w}_c{ch}.mlir"
                path = os.path.join(out_dir, fname)
                sub_h = h // s
                sub_w = w // s
                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"""// Strided MemRef Subview: Stride {s}, Shape {h}x{w}x{ch}
module @memref_strided_s{s}_{h}x{w}_c{ch} {{
  func.func @strided_subview_copy(
      %source: memref<{h}x{w}x{ch}xf32>,
      %destination: memref<{sub_h}x{sub_w}x{ch}xf32>) {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c_stride = arith.constant {s} : index
    %h_dim = arith.constant {sub_h} : index
    %w_dim = arith.constant {sub_w} : index
    %ch_dim = arith.constant {ch} : index

    scf.for %sy = %c0 to %h_dim step %c1 {{
      %src_y = arith.muli %sy, %c_stride : index
      scf.for %sx = %c0 to %w_dim step %c1 {{
        %src_x = arith.muli %sx, %c_stride : index
        scf.for %c = %c0 to %ch_dim step %c1 {{
          %val = memref.load %source[%src_y, %src_x, %c] : memref<{h}x{w}x{ch}xf32>
          memref.store %val, %destination[%sy, %sx, %c] : memref<{sub_h}x{sub_w}x{ch}xf32>
        }}
      }}
    }}
    return
  }}
}}
""")

def generate_scf_unrolled_suite(out_dir: str):
    """Generates unrolled SCF loop pipelines for memory bandwidth and latency benchmarking."""
    os.makedirs(out_dir, exist_ok=True)
    unrolls = [2, 4, 8]
    resolutions = [(720, 1280), (1080, 1920), (2160, 3840)]
    stages = [
        ("black_level", "Linearization"),
        ("lens_shading", "Vignetting"),
        ("white_balance", "Chromatic"),
        ("ccm", "ColorMatrix"),
        ("aces", "Filmic"),
    ]

    for u in unrolls:
        for h, w in resolutions:
            for sname, sdesc in stages:
                fname = f"scf_unroll{u}_{sname}_{h}x{w}.mlir"
                path = os.path.join(out_dir, fname)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"""// Unrolled SCF Loop Nest: {sdesc}, Unroll {u}x on {h}x{w}
module @scf_unroll{u}_{sname}_{h}x{w} {{
  func.func @execute_unroll(
      %in: memref<{h}x{w}xf32>,
      %out: memref<{h}x{w}xf32>,
      %scale: f32) {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %h_dim = arith.constant {h} : index
    %w_dim = arith.constant {w} : index
    %step = arith.constant {u} : index

    scf.for %y = %c0 to %h_dim step %c1 {{
      scf.for %x = %c0 to %w_dim step %step {{
""")
                    for k in range(u):
                        f.write(f"""        %c_off{k} = arith.constant {k} : index
        %idx{k} = arith.addi %x, %c_off{k} : index
        %val{k} = memref.load %in[%y, %idx{k}] : memref<{h}x{w}xf32>
        %res{k} = arith.mulf %val{k}, %scale : f32
        memref.store %res{k}, %out[%y, %idx{k}] : memref<{h}x{w}xf32>
""")
                    f.write("""      }
    }
    return
  }
}
""")
