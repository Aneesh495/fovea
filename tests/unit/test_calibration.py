"""
Unit tests for sensor calibration algorithms: black level, lens shading, and defective pixels.
"""
import unittest
from fovea.io.raw import RawBayerImage
from fovea.isp.calibration import (
    apply_black_level_subtraction,
    apply_lens_shading_correction,
    apply_defective_pixel_correction
)

class TestCalibration(unittest.TestCase):
    def test_black_level_subtraction(self):
        w, h = 4, 4
        # Range 64 to 1024
        data = [64.0, 304.0, 544.0, 1024.0] * 4
        img = RawBayerImage(w, h, "RGGB", 10, 64.0, 1024.0, data)
        lin = apply_black_level_subtraction(img, 64.0, 1024.0)

        self.assertEqual(len(lin), 16)
        self.assertAlmostEqual(lin[0], 0.0)
        self.assertAlmostEqual(lin[1], 0.25)
        self.assertAlmostEqual(lin[2], 0.50)
        self.assertAlmostEqual(lin[3], 1.0)

    def test_lens_shading_correction(self):
        w, h = 64, 64
        # Uniform field with optical falloff
        uniform = [0.5] * (w * h)
        # Apply vignetting gain polynomial: k0=1.0, k1=0.5 (corners boosted by up to 1.5x)
        corrected = apply_lens_shading_correction(uniform, w, h, radial_poly=[1.0, 0.5, 0.0, 0.0])

        center_idx = 32 * w + 32
        corner_idx = 0
        # Center gain is 1.0 (0.5 * 1.0 = 0.5)
        self.assertAlmostEqual(corrected[center_idx], 0.5, places=2)
        # Corner gain is ~1.5 (0.5 * 1.5 = 0.75)
        self.assertGreater(corrected[corner_idx], corrected[center_idx])

    def test_defective_pixel_correction(self):
        w, h = 8, 8
        # Flat field 0.4
        field = [0.4] * (w * h)
        # Inject one hot pixel at (4, 4)
        field[4 * w + 4] = 1.0
        # Inject one dead pixel at (4, 2)
        field[2 * w + 4] = 0.0

        repaired = apply_defective_pixel_correction(field, w, h, threshold=0.2)
        self.assertAlmostEqual(repaired[4 * w + 4], 0.4)
        self.assertAlmostEqual(repaired[2 * w + 4], 0.4)

if __name__ == "__main__":
    unittest.main()
