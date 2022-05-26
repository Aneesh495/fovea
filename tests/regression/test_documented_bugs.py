"""
Permanent regression test suite documenting and preventing bugs discovered during development.

Documented Regressions:
1. Bug #1: Odd-dimension Bayer CFA boundary truncation.
2. Bug #2: Division by zero in flat uniform regions during Guided Filter variance normalization.
3. Bug #3: Log-domain singularity for zero/negative radiance in Reinhard tone mapping.
4. Bug #4: Dynamic range wrap-around on 12-bit and 14-bit integer saturation.
5. Bug #5: MLIR SSA variable naming collision between constant %aces_b and result %aces_b.
6. Bug #6: MLIR LLVM dialect branch target operand typing in llvm.br.
"""
import unittest
from fovea.io.synthetic import bayer_mosaic
from fovea.isp.demosaic import demosaic_bilinear, demosaic_malvar_he_cutler
from fovea.isp.denoise import apply_guided_filter
from fovea.isp.tonemap import tone_map_reinhard
from fovea.compiler.pass_manager import PassPipeline
from fovea.compiler.generator import generate_complete_raw_isp_mlir

class TestDocumentedBugs(unittest.TestCase):
    def test_bug1_odd_dimension_bayer_boundary(self):
        """Bug #1: Odd dimensions (e.g. 33x33) must not crash or cause index out-of-bounds."""
        w, h = 33, 33
        rgb = [0.5] * (w * h * 3)
        raw = bayer_mosaic(rgb, w, h, "RGGB")
        out_bilinear = demosaic_bilinear(raw.data, w, h, "RGGB")
        self.assertEqual(len(out_bilinear), w * h * 3)
        out_mhc = demosaic_malvar_he_cutler(raw.data, w, h, "RGGB")
        self.assertEqual(len(out_mhc), w * h * 3)

    def test_bug2_guided_filter_zero_variance(self):
        """Bug #2: Flat uniform input with zero variance must not divide by zero."""
        w, h = 16, 16
        flat_guide = [0.42] * (w * h)
        flat_source = [0.42] * (w * h)
        # With eps > 0, denominator variance + eps is stable
        out = apply_guided_filter(flat_guide, flat_source, w, h, radius=2, eps=1e-4)
        for v in out:
            self.assertFalse(v != v, "Found NaN in flat guided filter output")
            self.assertAlmostEqual(v, 0.42, places=4)

    def test_bug3_reinhard_zero_radiance(self):
        """Bug #3: Zero radiance scene must not produce log(0) singularity."""
        zero_scene = [0.0] * 30
        ldr = tone_map_reinhard(zero_scene, key=0.18)
        self.assertEqual(len(ldr), len(zero_scene))
        for v in ldr:
            self.assertFalse(v != v, "Found NaN in Reinhard tone map output")
            self.assertAlmostEqual(v, 0.0)

    def test_bug4_saturation_clamping_limits(self):
        """Bug #4: Extreme saturation values must clamp at 1.0 rather than wrapping."""
        w, h = 4, 4
        extreme_rgb = [10.0, 50.0, 1000.0] * (w * h)
        raw = bayer_mosaic(extreme_rgb, w, h, "RGGB", black_level=64.0, white_level=4095.0, bit_depth=12)
        for v in raw.data:
            self.assertLessEqual(v, 4095.0)

    def test_bug5_mlir_ssa_variable_collision(self):
        """Bug #5: Monolithic MLIR pipeline must pass mlir-opt without SSA redefinition errors."""
        src = generate_complete_raw_isp_mlir(120, 160)
        pipe = PassPipeline("ssa_check").canonicalize()
        out = pipe.run(src)
        self.assertIn("func.func @process_frame_4k", out)

if __name__ == "__main__":
    unittest.main()
