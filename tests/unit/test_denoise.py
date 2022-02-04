"""
Unit tests for edge-preserving denoising: Bilateral and Guided filters.
"""
import unittest
from fovea.isp.denoise import apply_bilateral_filter, apply_guided_filter

class TestDenoise(unittest.TestCase):
    def test_bilateral_step_edge_preservation(self):
        # A sharp step edge image: left half 0.1, right half 0.9
        w, h = 16, 16
        img = [0.1 if (x < 8) else 0.9 for y in range(h) for x in range(w)]
        # Inject small noise
        noisy = list(img)
        noisy[4 * w + 4] += 0.05
        noisy[4 * w + 12] -= 0.05

        denoised = apply_bilateral_filter(noisy, w, h, channels=1, radius=2, sigma_spatial=2.0, sigma_range=0.1)

        # Left pixel at (4, 4) should remain close to 0.1, not blurred into 0.9
        self.assertLess(denoised[4 * w + 4], 0.2)
        # Right pixel at (4, 12) should remain close to 0.9, not blurred into 0.1
        self.assertGreater(denoised[4 * w + 12], 0.8)

    def test_guided_filter_edge_transfer(self):
        w, h = 12, 12
        # Guide with step edge
        guide = [0.2 if (x < 6) else 0.8 for y in range(h) for x in range(w)]
        source = [0.5] * (w * h)
        # Inject noise into source
        for i in range(len(source)):
            if i % 2 == 0:
                source[i] += 0.04
            else:
                source[i] -= 0.04

        filtered = apply_guided_filter(guide, source, w, h, radius=2, eps=1e-3)
        self.assertEqual(len(filtered), w * h)
        for v in filtered:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

if __name__ == "__main__":
    unittest.main()
