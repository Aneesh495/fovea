"""
End-to-end integration tests for full ISP pipeline.
Processes synthetic ColorChecker frames through all calibration, demosaic,
color correction, filtering, and tone reproduction stages.
"""
import unittest
from fovea.io.synthetic import generate_color_checker_chart, bayer_mosaic, add_sensor_noise
from fovea.isp.pipeline import ISPPipeline, ISPPipelineConfig

class TestEndToEndISP(unittest.TestCase):
    def test_full_pipeline_color_checker(self):
        w, h = 60, 40
        ground_truth = generate_color_checker_chart(w, h)
        raw = bayer_mosaic(ground_truth, w, h, "RGGB", black_level=64.0, white_level=4095.0, bit_depth=12)
        raw_noisy = add_sensor_noise(raw, read_noise_std=1.0, shot_noise_factor=0.2, num_hot_pixels=4, num_dead_pixels=4)

        config = ISPPipelineConfig(
            demosaic_algorithm="malvar_he_cutler",
            enable_lens_shading=True,
            enable_dpc=True,
            enable_denoise=True,
            enable_sharpen=True,
            tonemap_operator="aces"
        )
        pipeline = ISPPipeline(config)
        rgb_out = pipeline.process(raw_noisy)

        self.assertEqual(len(rgb_out), w * h * 3)
        for v in rgb_out:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

if __name__ == "__main__":
    unittest.main()
