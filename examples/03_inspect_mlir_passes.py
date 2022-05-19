#!/usr/bin/env python3
"""
Example 03: Inspecting Progressive MLIR Lowering Passes.
Demonstrates lowering high-level Affine and Vector dialect representations down to LLVM dialect.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fovea.compiler.lowering import generate_vectorized_white_balance, generate_tiled_affine_color_correction
from fovea.compiler.pass_manager import PassPipeline

def main():
    print("Fovea Example 03: Inspecting MLIR Lowering Passes")

    print("\n--- [Pass 1] Polyhedral Affine Tiled IR ---")
    affine_src = generate_tiled_affine_color_correction(640, 480, tile_h=32, tile_w=32)
    pipe_affine = PassPipeline("affine_opt").canonicalize().cse()
    opt_affine = pipe_affine.run(affine_src)
    print(opt_affine[:400] + "\n... [truncated] ...\n")

    print("\n--- [Pass 2] SIMD Vector Dialect to LLVM Dialect Lowering ---")
    vec_src = generate_vectorized_white_balance(1024, vector_width=8)
    pipe_llvm = PassPipeline("llvm_opt").lower_to_llvm()
    llvm_dialect = pipe_llvm.run(vec_src)
    print(llvm_dialect[:400] + "\n... [truncated] ...\n")

if __name__ == "__main__":
    main()
