#!/usr/bin/env python3
"""
Example 04: JIT Compilation of Custom MLIR Kernels.
Compiles a vectorized vector.broadcast and arith.mulf kernel into native machine code.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fovea.compiler.jit import MLIRJITEngine
from fovea.compiler.lowering import generate_vectorized_white_balance

def main():
    print("Fovea Example 04: Native JIT Compilation")
    engine = MLIRJITEngine(opt_level="-O3")

    src = generate_vectorized_white_balance(2048, vector_width=16)
    print("Compiling 16-wide SIMD vector kernel to native shared library...")
    dylib = engine.compile_module_to_dylib(src)
    print(f"JIT Compilation SUCCESS: Emitted native binary at {dylib}")

    if os.path.exists(dylib):
        os.remove(dylib)

if __name__ == "__main__":
    main()
