"""
Pass manager for orchestrating MLIR optimization and lowering pipelines.
Executes canonicalization, polyhedral affine tiling, SIMD vectorization, and LLVM lowering.
"""
import os
import subprocess
from typing import List, Optional

MLIR_OPT_PATHS = [
    "/opt/homebrew/opt/llvm/bin/mlir-opt",
    "/usr/local/opt/llvm/bin/mlir-opt",
    "mlir-opt"
]

def find_mlir_opt() -> str:
    for p in MLIR_OPT_PATHS:
        if os.path.exists(p):
            return p
    return "mlir-opt"

class PassPipeline:
    """Configures a sequence of MLIR passes."""

    def __init__(self, name: str = "custom_pipeline"):
        self.name = name
        self.passes: List[str] = []

    def add_pass(self, pass_name: str) -> 'PassPipeline':
        self.passes.append(pass_name)
        return self

    def canonicalize(self) -> 'PassPipeline':
        return self.add_pass("--canonicalize")

    def cse(self) -> 'PassPipeline':
        return self.add_pass("--cse")

    def affine_loop_tile(self, tile_size: int = 32) -> 'PassPipeline':
        return self.add_pass(f"--affine-loop-tile=tile-size={tile_size}")

    def affine_loop_unroll(self, factor: int = 4) -> 'PassPipeline':
        return self.add_pass(f"--affine-loop-unroll=unroll-factor={factor}")

    def lower_affine_to_scf(self) -> 'PassPipeline':
        return self.add_pass("--lower-affine")

    def lower_vector_to_scf(self) -> 'PassPipeline':
        return self.add_pass("--convert-vector-to-scf")

    def lower_scf_to_cf(self) -> 'PassPipeline':
        return self.add_pass("--convert-scf-to-cf")

    def lower_to_llvm(self) -> 'PassPipeline':
        return (
            self.add_pass("--convert-vector-to-scf")
                .add_pass("--convert-scf-to-cf")
                .add_pass("--convert-cf-to-llvm")
                .add_pass("--convert-vector-to-llvm")
                .add_pass("--convert-arith-to-llvm")
                .add_pass("--convert-math-to-llvm")
                .add_pass("--convert-func-to-llvm")
                .add_pass("--finalize-memref-to-llvm")
                .add_pass("--reconcile-unrealized-casts")
        )

    def run(self, mlir_source: str) -> str:
        """Executes the pass pipeline using mlir-opt and returns optimized MLIR text."""
        mlir_opt = find_mlir_opt()
        cmd = [mlir_opt] + self.passes
        res = subprocess.run(
            cmd,
            input=mlir_source,
            capture_output=True,
            text=True
        )
        if res.returncode != 0:
            raise RuntimeError(f"Pass pipeline failed (exit {res.returncode}):\n{res.stderr}")
        return res.stdout
