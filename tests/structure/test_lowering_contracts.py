"""
Structural verification tests for progressive lowering pipelines.
Validates Affine tiling, Vector SIMD lowering, and LLVM dialect conversion.
"""
import unittest
from fovea.compiler.lowering import (
    generate_tiled_affine_color_correction,
    generate_vectorized_white_balance
)
from fovea.compiler.pass_manager import PassPipeline

class TestLoweringContracts(unittest.TestCase):
    def test_affine_tiling_pipeline(self):
        src = generate_tiled_affine_color_correction(1920, 1080, 32, 32)
        pipe = PassPipeline("affine_test").canonicalize().cse()
        out = pipe.run(src)
        self.assertIn("affine.for", out)
        self.assertIn("affine.load", out)
        self.assertIn("affine.store", out)

    def test_vector_simd_to_llvm_pipeline(self):
        src = generate_vectorized_white_balance(1024, vector_width=8)
        pipe = PassPipeline("vector_test").lower_to_llvm()
        out = pipe.run(src)
        self.assertIn("llvm.func", out)
        self.assertIn("llvm.ptr", out)

if __name__ == "__main__":
    unittest.main()
