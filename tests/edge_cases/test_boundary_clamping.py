"""
Edge case and stress tests: boundary conditions, non-square aspects, and extreme exposures.
"""
import unittest
from fovea.io.synthetic import generate_color_checker_chart, bayer_mosaic, generate_exposure_bracket
from fovea.isp.pipeline import ISPPipeline, ISPPipelineConfig

class TestBoundaryClamping(unittest.TestCase):
    def test_extreme_aspect_ratio_ultrawide(self):
        """Ultra-wide aspect ratio (e.g. panoramic 200x20)."""
        w, h = 200, 20
        gt = generate_color_checker_chart(w, h)
        raw = bayer_mosaic(gt, w, h, "RGGB")
        pipeline = ISPPipeline()
        out = pipeline.process(raw)
        self.assertEqual(len(out), w * h * 3)

    def test_extreme_aspect_ratio_tall(self):
        """Tall aspect ratio (e.g. 20x200)."""
        w, h = 20, 200
        gt = generate_color_checker_chart(w, h)
        raw = bayer_mosaic(gt, w, h, "RGGB")
        pipeline = ISPPipeline()
        out = pipeline.process(raw)
        self.assertEqual(len(out), w * h * 3)

    def test_extreme_ev_brackets(self):
        """Extreme EV bracket ranges from -10.0 EV to +10.0 EV."""
        w, h = 16, 16
        rgb = [0.1, 0.5, 0.9] * (w * h)
        brackets = generate_exposure_bracket(rgb, w, h, ev_stops=[-10.0, 0.0, 10.0])
        self.assertEqual(len(brackets), 3)
        for b in brackets:
            for v in b:
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)

if __name__ == "__main__":
    unittest.main()
