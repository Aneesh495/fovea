"""
Unit tests for multi-exposure HDR fusion.
"""
import unittest
from fovea.io.synthetic import generate_exposure_bracket
from fovea.isp.hdr import fuse_exposure_brackets

class TestHDR(unittest.TestCase):
    def test_exposure_fusion_dynamic_range(self):
        w, h = 8, 8
        # High dynamic range scene with dark shadows and bright highlights
        base_scene = []
        for y in range(h):
            for x in range(w):
                val = 0.05 if x < 4 else 0.95
                base_scene.extend([val, val, val])

        brackets = generate_exposure_bracket(base_scene, w, h, ev_stops=[-2.0, 0.0, 2.0])
        fused = fuse_exposure_brackets(brackets, w, h)

        self.assertEqual(len(fused), w * h * 3)
        # Fused values should be well balanced in [0.0, 1.0]
        for v in fused:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

if __name__ == "__main__":
    unittest.main()
