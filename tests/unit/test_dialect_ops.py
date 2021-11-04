"""
Unit tests for Fovea IR building and text emitting.
"""
import unittest
from fovea.compiler.ir import Module, Function, Block, Value, TensorType, F32Type
from fovea.compiler.emitter import MLIREmitter
from fovea.compiler.dialect import FoveaDialect

class TestDialectOps(unittest.TestCase):
    def test_emit_fovea_pipeline_ir(self):
        """Build an end-to-end AST of an ISP pipeline and verify emitted MLIR."""
        mod = Module(name="test_isp")
        tensor_2d = TensorType([1080, 1920], F32Type())
        tensor_3d = TensorType([1080, 1920, 3], F32Type())

        fn = Function("process_frame", arg_types=[tensor_2d], ret_types=[tensor_3d])
        block = Block("entry")

        arg0 = Value("arg0", tensor_2d)

        # 1. Black level
        op_bl = FoveaDialect.black_level(arg0, offset=64.0, saturation=1023.0, result_name="lin")
        block.add_operation(op_bl)

        # 2. Lens shading
        op_ls = FoveaDialect.lens_shading(op_bl.results[0], [1.0, 0.05, 0.01, 0.0], [960.0, 540.0], result_name="shading")
        block.add_operation(op_ls)

        # 3. Demosaic
        op_dm = FoveaDialect.demosaic(op_ls.results[0], cfa_pattern="RGGB", algorithm="malvar_he_cutler", result_name="rgb")
        block.add_operation(op_dm)

        # 4. White balance
        op_wb = FoveaDialect.white_balance(op_dm.results[0], gain_r=2.1, gain_g=1.0, gain_b=1.8, result_name="wb")
        block.add_operation(op_wb)

        # 5. Tone map
        op_tm = FoveaDialect.tone_map(op_wb.results[0], operator_type="aces", key_value=0.18, result_name="aces")
        block.add_operation(op_tm)

        # 6. Gamma
        op_gm = FoveaDialect.gamma(op_tm.results[0], gamma_value=2.2, result_name="out")
        block.add_operation(op_gm)

        fn.region.add_block(block)
        mod.add_function(fn)

        emitter = MLIREmitter()
        mlir_text = emitter.emit_module(mod)

        self.assertIn("module @test_isp", mlir_text)
        self.assertIn("fovea.black_level", mlir_text)
        self.assertIn("fovea.lens_shading", mlir_text)
        self.assertIn("fovea.demosaic", mlir_text)
        self.assertIn("fovea.white_balance", mlir_text)
        self.assertIn("fovea.tone_map", mlir_text)
        self.assertIn("fovea.gamma", mlir_text)
        self.assertIn("tensor<1080x1920xf32>", mlir_text)
        self.assertIn("tensor<1080x1920x3xf32>", mlir_text)

if __name__ == "__main__":
    unittest.main()
