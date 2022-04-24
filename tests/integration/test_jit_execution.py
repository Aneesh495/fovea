"""
Integration tests for MLIR JIT compilation and dynamic execution.
Verifies that lowered MLIR kernels compile to native machine code libraries.
"""
import os
import unittest
from fovea.compiler.jit import MLIRJITEngine
from fovea.compiler.lowering import generate_vectorized_white_balance

class TestJITExecution(unittest.TestCase):
    def test_jit_compile_vector_kernel(self):
        engine = MLIRJITEngine(opt_level="-O2")
        src = generate_vectorized_white_balance(1024, vector_width=8)
        dylib_path = engine.compile_module_to_dylib(src)

        try:
            self.assertTrue(os.path.exists(dylib_path))
            self.assertGreater(os.path.getsize(dylib_path), 0)
        finally:
            if os.path.exists(dylib_path):
                os.remove(dylib_path)

if __name__ == "__main__":
    unittest.main()
