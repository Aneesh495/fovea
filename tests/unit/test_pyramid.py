"""
Unit tests for Gaussian and Laplacian multi-scale image pyramids.
Verifies dyadic decimation, expansion, and pyramid reconstruction accuracy.
"""
import unittest
from fovea.isp.pyramid import (
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    reconstruct_laplacian_pyramid
)

class TestPyramid(unittest.TestCase):
    def test_gaussian_pyramid_hierarchy(self):
        w, h = 64, 48
        img = [float((x + y) % 10) / 10.0 for y in range(h) for x in range(w)]
        pyr = build_gaussian_pyramid(img, w, h, levels=4)

        self.assertEqual(len(pyr), 4)
        self.assertEqual((pyr[0][1], pyr[0][2]), (64, 48))
        self.assertEqual((pyr[1][1], pyr[1][2]), (32, 24))
        self.assertEqual((pyr[2][1], pyr[2][2]), (16, 12))
        self.assertEqual((pyr[3][1], pyr[3][2]), (8, 6))

    def test_laplacian_pyramid_reconstruction(self):
        w, h = 32, 32
        img = [float((x * 3 + y * 7) % 256) / 255.0 for y in range(h) for x in range(w)]
        pyr = build_laplacian_pyramid(img, w, h, levels=3)
        reconstructed = reconstruct_laplacian_pyramid(pyr)

        self.assertEqual(len(reconstructed), len(img))
        # Reconstruction error across the image should be virtually identical
        max_err = max(abs(a - b) for a, b in zip(img, reconstructed))
        self.assertLess(max_err, 0.05)

if __name__ == "__main__":
    unittest.main()
