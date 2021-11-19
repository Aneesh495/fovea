"""
Unit tests for synthetic calibration targets and physical sensor noise models.
"""
import unittest
from fovea.io.synthetic import (
    generate_color_checker_chart,
    generate_siemens_star,
    bayer_mosaic,
    add_sensor_noise,
    generate_exposure_bracket
)

class TestSynthetic(unittest.TestCase):
    def test_color_checker_dimensions(self):
        w, h = 120, 80
        chart = generate_color_checker_chart(w, h)
        self.assertEqual(len(chart), w * h * 3)
        for v in chart:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_siemens_star_generation(self):
        w, h = 64, 64
        star = generate_siemens_star(w, h, spokes=16)
        self.assertEqual(len(star), w * h * 3)
        center_idx = (32 * w + 32) * 3
        self.assertTrue(0.0 <= star[center_idx] <= 1.0)

    def test_bayer_mosaic_patterns(self):
        w, h = 8, 8
        rgb = [1.0, 0.5, 0.25] * (w * h)
        for pattern in ["RGGB", "BGGR", "GRBG", "GBRG"]:
            raw = bayer_mosaic(rgb, w, h, cfa_pattern=pattern, black_level=64.0, white_level=1023.0)
            self.assertEqual(raw.width, w)
            self.assertEqual(raw.height, h)
            self.assertEqual(raw.cfa_pattern, pattern)
            self.assertEqual(len(raw.data), w * h)
            for v in raw.data:
                self.assertGreaterEqual(v, 64.0)
                self.assertLessEqual(v, 1023.0)

    def test_noise_and_defective_pixel_injection(self):
        w, h = 32, 32
        rgb = [0.5, 0.5, 0.5] * (w * h)
        raw = bayer_mosaic(rgb, w, h, black_level=64.0, white_level=4095.0)
        noisy = add_sensor_noise(raw, num_hot_pixels=5, num_dead_pixels=5, seed=123)
        self.assertEqual(len(noisy.data), w * h)

        hot_count = sum(1 for v in noisy.data if v == 4095.0)
        dead_count = sum(1 for v in noisy.data if v == 64.0)
        self.assertGreaterEqual(hot_count, 5)
        self.assertGreaterEqual(dead_count, 5)

    def test_exposure_bracketing(self):
        w, h = 4, 4
        rgb = [0.5] * (w * h * 3)
        brackets = generate_exposure_bracket(rgb, w, h, ev_stops=[-1.0, 0.0, 1.0])
        self.assertEqual(len(brackets), 3)
        # EV -1 should be 0.25, EV 0 should be 0.5, EV +1 should be 1.0
        self.assertAlmostEqual(brackets[0][0], 0.25)
        self.assertAlmostEqual(brackets[1][0], 0.50)
        self.assertAlmostEqual(brackets[2][0], 1.00)

if __name__ == "__main__":
    unittest.main()
