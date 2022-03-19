"""
Unit tests for tone mapping operators: Reinhard, Drago, and ACES Filmic.
"""
import unittest
from fovea.isp.tonemap import tone_map_reinhard, tone_map_drago, tone_map_aces

class TestTonemap(unittest.TestCase):
    def test_reinhard_compression(self):
        # HDR pixels with extreme luminance
        hdr = [0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 100.0, 100.0, 100.0]
        ldr = tone_map_reinhard(hdr, key=0.18, white_point=10.0)

        self.assertEqual(len(ldr), len(hdr))
        for v in ldr:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)
        # Monotonicity: brighter input should produce brighter mapped output
        self.assertLess(ldr[0], ldr[3])
        self.assertLess(ldr[3], ldr[6])
        self.assertLess(ldr[6], ldr[9])

    def test_drago_compression(self):
        hdr = [0.05, 0.05, 0.05, 2.0, 2.0, 2.0, 50.0, 50.0, 50.0]
        ldr = tone_map_drago(hdr, bias=0.85)
        self.assertEqual(len(ldr), len(hdr))
        for v in ldr:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_aces_filmic_curve(self):
        # Test toe (near black), midtone, and highlight rolloff
        samples = [0.0, 0.18, 1.0, 10.0]
        mapped = tone_map_aces(samples)

        # 0.0 should remain 0.0
        self.assertAlmostEqual(mapped[0], 0.0)
        # 0.18 should map to ACES curve value ~0.2669
        self.assertGreater(mapped[1], 0.20)
        self.assertLess(mapped[1], 0.35)
        # 10.0 should roll off cleanly to highlight saturation (1.0)
        self.assertLessEqual(mapped[3], 1.0)
        self.assertGreaterEqual(mapped[3], 0.95)

if __name__ == "__main__":
    unittest.main()
