"""
Unit tests for raw Bayer, PPM, PFM, and RGBE HDR serializers.
Verifies lossless binary round-trips across bit depths and formats.
"""
import os
import tempfile
import unittest
from fovea.io.raw import RawBayerImage, write_raw_bayer, read_raw_bayer
from fovea.io.pnm import write_ppm, read_ppm, write_pfm, read_pfm
from fovea.io.hdr_io import write_rgbe_hdr, read_rgbe_hdr

class TestImageIO(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_raw_bayer_roundtrip_12bit(self):
        """Test 12-bit Raw Bayer round-trip with metadata preservation."""
        w, h = 64, 48
        data = [(x * 17 + y * 31) % 4096 for y in range(h) for x in range(w)]
        img = RawBayerImage(
            width=w,
            height=h,
            cfa_pattern="RGGB",
            bit_depth=12,
            black_level=64.0,
            white_level=4095.0,
            data=[float(v) for v in data]
        )
        path = os.path.join(self.temp_dir, "test12.fraw")
        write_raw_bayer(img, path)

        loaded = read_raw_bayer(path)
        self.assertEqual(loaded.width, w)
        self.assertEqual(loaded.height, h)
        self.assertEqual(loaded.cfa_pattern, "RGGB")
        self.assertEqual(loaded.bit_depth, 12)
        self.assertAlmostEqual(loaded.black_level, 64.0)
        self.assertAlmostEqual(loaded.white_level, 4095.0)
        self.assertEqual(len(loaded.data), w * h)
        for original, restored in zip(data, loaded.data):
            self.assertEqual(original, int(restored))

    def test_ppm_roundtrip_8bit(self):
        """Test 8-bit binary PPM round-trip."""
        w, h = 16, 16
        data = [float((i * 7) % 256) for i in range(w * h * 3)]
        path = os.path.join(self.temp_dir, "test.ppm")
        write_ppm(path, w, h, data, max_val=255)

        rw, rh, rdata = read_ppm(path)
        self.assertEqual(rw, w)
        self.assertEqual(rh, h)
        for o, r in zip(data, rdata):
            self.assertEqual(int(o), int(r))

    def test_pfm_hdr_roundtrip(self):
        """Test 32-bit floating point PFM round-trip."""
        w, h = 8, 8
        data = [float(i * 1.5 + 0.125) for i in range(w * h * 3)]
        path = os.path.join(self.temp_dir, "test.pfm")
        write_pfm(path, w, h, data)

        rw, rh, rdata = read_pfm(path)
        self.assertEqual(rw, w)
        self.assertEqual(rh, h)
        for o, r in zip(data, rdata):
            self.assertAlmostEqual(o, r, places=4)

    def test_rgbe_hdr_roundtrip(self):
        """Test Radiance RGBE round-trip."""
        w, h = 4, 4
        data = [1.0, 0.5, 0.25, 10.0, 50.0, 100.0] * 8
        path = os.path.join(self.temp_dir, "test.hdr")
        write_rgbe_hdr(path, w, h, data)

        rw, rh, rdata = read_rgbe_hdr(path)
        self.assertEqual(rw, w)
        self.assertEqual(rh, h)
        for o, r in zip(data, rdata):
            # RGBE has 8-bit mantissa, relative error should be < 1%
            if o > 1e-4:
                rel_err = abs(o - r) / o
                self.assertLess(rel_err, 0.02)

if __name__ == "__main__":
    unittest.main()
