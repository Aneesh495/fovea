"""
Unit tests for CFA demosaicing algorithms.
Validates Bilinear, Malvar-He-Cutler, and Directional gradient demosaicing.
"""
import unittest
from fovea.io.synthetic import bayer_mosaic, generate_color_checker_chart
from fovea.isp.demosaic import (
    demosaic_bilinear,
    demosaic_malvar_he_cutler,
    demosaic_directional
)

class TestDemosaic(unittest.TestCase):
    def setUp(self):
        self.w, self.h = 60, 40
        self.ground_truth = generate_color_checker_chart(self.w, self.h)
        self.raw = bayer_mosaic(
            self.ground_truth,
            self.w,
            self.h,
            cfa_pattern="RGGB",
            black_level=0.0,
            white_level=1.0,
            bit_depth=16
        )

    def test_demosaic_bilinear_reconstruction(self):
        rgb = demosaic_bilinear(self.raw.data, self.w, self.h, "RGGB")
        self.assertEqual(len(rgb), self.w * self.h * 3)

        # Check interior pixels for reasonable PSNR
        diffs = []
        for y in range(4, self.h - 4):
            for x in range(4, self.w - 4):
                idx = (y * self.w + x) * 3
                for c in range(3):
                    diffs.append((rgb[idx + c] - self.ground_truth[idx + c]) ** 2)
        mse = sum(diffs) / len(diffs)
        self.assertLess(mse, 0.05)

    def test_demosaic_malvar_he_cutler_reconstruction(self):
        rgb = demosaic_malvar_he_cutler(self.raw.data, self.w, self.h, "RGGB")
        self.assertEqual(len(rgb), self.w * self.h * 3)

        diffs = []
        for y in range(4, self.h - 4):
            for x in range(4, self.w - 4):
                idx = (y * self.w + x) * 3
                for c in range(3):
                    diffs.append((rgb[idx + c] - self.ground_truth[idx + c]) ** 2)
        mse = sum(diffs) / len(diffs)
        self.assertLess(mse, 0.04)

    def test_demosaic_directional_reconstruction(self):
        rgb = demosaic_directional(self.raw.data, self.w, self.h, "RGGB")
        self.assertEqual(len(rgb), self.w * self.h * 3)

        diffs = []
        for y in range(4, self.h - 4):
            for x in range(4, self.w - 4):
                idx = (y * self.w + x) * 3
                for c in range(3):
                    diffs.append((rgb[idx + c] - self.ground_truth[idx + c]) ** 2)
        mse = sum(diffs) / len(diffs)
        self.assertLess(mse, 0.05)

if __name__ == "__main__":
    unittest.main()
