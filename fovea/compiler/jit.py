"""
Just-In-Time (JIT) execution engine for MLIR pipelines.
Lowers MLIR modules to LLVM IR, compiles to native machine code via Clang, and invokes via ctypes.
"""
import ctypes
import os
import subprocess
import tempfile
from typing import List, Optional
from .pass_manager import PassPipeline, find_mlir_opt

MLIR_TRANSLATE_PATHS = [
    "/opt/homebrew/opt/llvm/bin/mlir-translate",
    "/usr/local/opt/llvm/bin/mlir-translate",
    "mlir-translate"
]

CLANG_PATHS = [
    "/opt/homebrew/opt/llvm/bin/clang",
    "/usr/local/opt/llvm/bin/clang",
    "clang"
]

def find_binary(candidates: List[str]) -> str:
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[-1]

class MLIRJITEngine:
    """Compiles MLIR source into native machine code dynamic library."""

    def __init__(self, opt_level: str = "-O3"):
        self.opt_level = opt_level
        self.mlir_opt = find_mlir_opt()
        self.mlir_translate = find_binary(MLIR_TRANSLATE_PATHS)
        self.clang = find_binary(CLANG_PATHS)

    def compile_module_to_dylib(self, mlir_src: str, output_dylib: Optional[str] = None) -> str:
        """Translates MLIR source to LLVM dialect, LLVM IR, and compiles to shared library."""
        # 1. Lower to LLVM dialect in MLIR
        pipeline = PassPipeline("jit_lowering").lower_to_llvm()
        llvm_dialect_mlir = pipeline.run(mlir_src)

        # 2. Translate MLIR LLVM dialect to LLVM IR (.ll)
        res_trans = subprocess.run(
            [self.mlir_translate, "--mlir-to-llvmir"],
            input=llvm_dialect_mlir,
            capture_output=True,
            text=True
        )
        if res_trans.returncode != 0:
            raise RuntimeError(f"mlir-translate failed: {res_trans.stderr}")

        llvm_ir = res_trans.stdout

        # 3. Write LLVM IR to temporary file and compile with Clang
        if output_dylib is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".dylib", delete=False)
            output_dylib = tmp.name
            tmp.close()

        with tempfile.NamedTemporaryFile(suffix=".ll", mode="w", delete=False) as f:
            f.write(llvm_ir)
            ll_path = f.name

        try:
            res_clang = subprocess.run(
                [self.clang, self.opt_level, "-shared", ll_path, "-o", output_dylib],
                capture_output=True,
                text=True
            )
            if res_clang.returncode != 0:
                raise RuntimeError(f"Clang native compilation failed: {res_clang.stderr}")
        finally:
            if os.path.exists(ll_path):
                os.remove(ll_path)

        return output_dylib
