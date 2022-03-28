"""
Progressive lowering strategies from high-level Fovea ops to Affine, Vector, MemRef, and LLVM.
"""
from typing import List, Optional

def generate_tiled_affine_color_correction(
    width: int,
    height: int,
    tile_h: int = 32,
    tile_w: int = 32
) -> str:
    """Generates polyhedral tiled affine loop nests for 3x3 color correction matrix."""
    return f"""// Tiled Polyhedral Affine Loop Nest for 3x3 Color Correction Matrix
// Evaluates {tile_h}x{tile_w} cache blocks to guarantee L1 data cache residency.

#map_tile_h = affine_map<(d0) -> (d0 + {tile_h})>
#map_tile_w = affine_map<(d0) -> (d0 + {tile_w})>

module @tiled_color_correction {{
  func.func @ccm_affine_tiled(
      %input: memref<{height}x{width}x3xf32>,
      %output: memref<{height}x{width}x3xf32>,
      %m00: f32, %m01: f32, %m02: f32,
      %m10: f32, %m11: f32, %m12: f32,
      %m20: f32, %m21: f32, %m22: f32) {{
    %f0 = arith.constant 0.000000e+00 : f32
    %f1 = arith.constant 1.000000e+00 : f32

    affine.for %ty = 0 to {height} step {tile_h} {{
      affine.for %tx = 0 to {width} step {tile_w} {{
        affine.for %y = affine_map<(d0) -> (d0)>(%ty) to min #map_tile_h(%ty) {{
          affine.for %x = affine_map<(d0) -> (d0)>(%tx) to min #map_tile_w(%tx) {{
            %r = affine.load %input[%y, %x, 0] : memref<{height}x{width}x3xf32>
            %g = affine.load %input[%y, %x, 1] : memref<{height}x{width}x3xf32>
            %b = affine.load %input[%y, %x, 2] : memref<{height}x{width}x3xf32>

            %r0 = arith.mulf %m00, %r : f32
            %r1 = arith.mulf %m01, %g : f32
            %r2 = arith.mulf %m02, %b : f32
            %r_sum = arith.addf %r0, %r1 : f32
            %r_out = arith.addf %r_sum, %r2 : f32

            %g0 = arith.mulf %m10, %r : f32
            %g1 = arith.mulf %m11, %g : f32
            %g2 = arith.mulf %m12, %b : f32
            %g_sum = arith.addf %g0, %g1 : f32
            %g_out = arith.addf %g_sum, %g2 : f32

            %b0 = arith.mulf %m20, %r : f32
            %b1 = arith.mulf %m21, %g : f32
            %b2 = arith.mulf %m22, %b : f32
            %b_sum = arith.addf %b0, %b1 : f32
            %b_out = arith.addf %b_sum, %b2 : f32

            affine.store %r_out, %output[%y, %x, 0] : memref<{height}x{width}x3xf32>
            affine.store %g_out, %output[%y, %x, 1] : memref<{height}x{width}x3xf32>
            affine.store %b_out, %output[%y, %x, 2] : memref<{height}x{width}x3xf32>
          }}
        }}
      }}
    }}
    return
  }}
}}
"""

def generate_vectorized_white_balance(
    total_pixels: int,
    vector_width: int = 8
) -> str:
    """Generates vectorized SIMD white balance kernel using vector.transfer_read and broadcast."""
    return f"""// Vectorized SIMD White Balance Kernel in MLIR
// Processes {vector_width} single-precision floats per vector register.

module @vectorized_white_balance {{
  func.func @wb_vector_{vector_width}x(
      %r_channel: memref<{total_pixels}xf32>,
      %r_out: memref<{total_pixels}xf32>,
      %gain_r: f32) {{
    %c0 = arith.constant 0 : index
    %c_total = arith.constant {total_pixels} : index
    %step = arith.constant {vector_width} : index
    %f0 = arith.constant 0.000000e+00 : f32

    %v_gain = vector.broadcast %gain_r : f32 to vector<{vector_width}xf32>

    scf.for %i = %c0 to %c_total step %step {{
      %v_in = vector.transfer_read %r_channel[%i], %f0 : memref<{total_pixels}xf32>, vector<{vector_width}xf32>
      %v_res = arith.mulf %v_in, %v_gain : vector<{vector_width}xf32>
      vector.transfer_write %v_res, %r_out[%i] : vector<{vector_width}xf32>, memref<{total_pixels}xf32>
    }}
    return
  }}
}}
"""
