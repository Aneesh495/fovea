"""
Unit tests for detail enhancement and unsharp masking.
"""
import unittest
from fovea.isp.enhancement import apply_unsharp_mask

class TestEnhancement(unittest.TestCase):
    def test_unsharp_mask_edge_boost(self):
        w, h = 16, 16
        # Step edge
        img = [0.2 if (x < 8) else 0.8 for y in range(h) for x in range(w)]
        sharpened = apply_unsharp_mask(img, w, h, channels=1, strength=0.8, threshold=0.01)

        # The transition pixel on the dark side should be pushed darker (< 0.2)
        self.assertLess(sharpened[8 * w + 7], 0.20)
        # The transition pixel on the bright side should be pushed brighter (> 0.8)
        self.assertGreater(sharpened[8 * w + 8], 0.80)

    def test_unsharp_mask_coring_threshold(self):
        w, h = 8, 8
        # Flat region with sub-threshold noise
        img = [0.5 + 0.005 * ((x + y) % 2) for y in range(h) for x in range(w)]
        # Threshold 0.02 is higher than noise amplitude 0.005
        sharpened = apply_unsharp_mask(img, w, h, channels=1, strength=1.0, threshold=0.02)

        # Output should be un-amplified (exact original)
        for orig, sharp in zip(img, sharpened):
            self.assertAlmostEqual(orig, sharp)

if __name__ == "__main__":
    unittest.main()
