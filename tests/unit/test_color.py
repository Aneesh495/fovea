"""
Unit tests for color science processing: white balance, CCM, and gamma curves.
"""
import unittest
from fovea.isp.color import (
    apply_white_balance,
    estimate_white_balance_gray_world,
    apply_color_correction_matrix,
    apply_gamma_curve,
    apply_srgb_eotf
)

class TestColor(unittest.TestCase):
    def test_white_balance_scaling(self):
        rgb = [0.2, 0.4, 0.1]
        wb = apply_white_balance(rgb, gain_r=2.0, gain_g=1.0, gain_b=3.0)
        self.assertAlmostEqual(wb[0], 0.4)
        self.assertAlmostEqual(wb[1], 0.4)
        self.assertAlmostEqual(wb[2], 0.3)

    def test_gray_world_estimation(self):
        # A warm reddish scene (high R, normal G, low B)
        rgb = [0.6, 0.3, 0.15] * 100
        gr, gg, gb = estimate_white_balance_gray_world(rgb)
        self.assertAlmostEqual(gg, 1.0)
        self.assertAlmostEqual(gr, 0.5)
        self.assertAlmostEqual(gb, 2.0)

    def test_color_correction_identity(self):
        identity_ccm = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ]
        rgb = [0.25, 0.5, 0.75]
        out = apply_color_correction_matrix(rgb, identity_ccm)
        self.assertAlmostEqual(out[0], 0.25)
        self.assertAlmostEqual(out[1], 0.50)
        self.assertAlmostEqual(out[2], 0.75)

    def test_srgb_eotf_toe_and_power(self):
        # Value in linear toe region
        toe_val = 0.001
        res_toe = apply_srgb_eotf([toe_val])
        self.assertAlmostEqual(res_toe[0], 12.92 * toe_val)

        # Mid-gray 0.18 should map to standard sRGB ~0.461356
        mid_val = 0.18
        res_mid = apply_srgb_eotf([mid_val])
        self.assertAlmostEqual(res_mid[0], 0.461356, places=4)

if __name__ == "__main__":
    unittest.main()
