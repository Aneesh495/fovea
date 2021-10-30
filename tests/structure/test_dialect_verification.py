"""
Structural and semantic verification tests for Fovea MLIR dialect definitions.
Verifies that all dialect operations parse correctly and satisfy IRDL constraints.
"""
import os
import subprocess
import unittest

MLIR_OPT_CANDIDATES = [
    "/opt/homebrew/opt/llvm/bin/mlir-opt",
    "/usr/local/opt/llvm/bin/mlir-opt",
    "mlir-opt"
]

def get_mlir_opt():
    for candidate in MLIR_OPT_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return "mlir-opt"

class TestDialectVerification(unittest.TestCase):
    def test_irdl_dialect_syntax(self):
        """Verify that fovea.irdl.mlir is syntactically valid and loads into mlir-opt."""
        mlir_opt = get_mlir_opt()
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        irdl_path = os.path.join(repo_root, "dialects", "irdl", "fovea.irdl.mlir")
        self.assertTrue(os.path.exists(irdl_path), f"IRDL file not found at {irdl_path}")

        cmd = [mlir_opt, irdl_path]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"mlir-opt failed to verify IRDL dialect:\n{res.stderr}")

    def test_tablegen_specifications_exist(self):
        """Verify that TableGen specification files exist and contain core operation declarations."""
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        td_path = os.path.join(repo_root, "dialects", "tablegen", "FoveaOps.td")
        self.assertTrue(os.path.exists(td_path))
        with open(td_path, "r", encoding="utf-8") as f:
            content = f.read()

        expected_ops = [
            "Fovea_BlackLevelOp",
            "Fovea_LensShadingOp",
            "Fovea_DefectivePixelOp",
            "Fovea_DemosaicOp",
            "Fovea_WhiteBalanceOp",
            "Fovea_ColorCorrectionOp",
            "Fovea_BilateralOp",
            "Fovea_GuidedFilterOp",
            "Fovea_UnsharpMaskOp",
            "Fovea_HdrFuseOp",
            "Fovea_PyrDownOp",
            "Fovea_PyrUpOp",
            "Fovea_ToneMapOp",
            "Fovea_GammaOp"
        ]
        for op in expected_ops:
            self.assertIn(op, content, f"Missing op definition: {op}")

if __name__ == "__main__":
    unittest.main()
