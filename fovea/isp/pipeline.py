"""
Modular Image Signal Processor (ISP) pipeline execution orchestrator.
Chains sensor calibration, demosaicing, white balance, color correction,
denoising, detail sharpening, and perceptual tone mapping into unified execution flow.
"""
from typing import Dict, List, Optional
from ..io.raw import RawBayerImage
from .calibration import (
    apply_black_level_subtraction,
    apply_lens_shading_correction,
    apply_defective_pixel_correction
)
from .demosaic import (
    demosaic_bilinear,
    demosaic_malvar_he_cutler,
    demosaic_directional
)
from .color import (
    apply_white_balance,
    apply_color_correction_matrix,
    apply_srgb_eotf,
    DEFAULT_CCM_SRGB
)
from .denoise import apply_bilateral_filter, apply_guided_filter
from .enhancement import apply_unsharp_mask
from .tonemap import tone_map_reinhard, tone_map_aces

class ISPPipelineConfig:
    """Configuration hyperparameters for all stages of the ISP pipeline."""

    def __init__(
        self,
        demosaic_algorithm: str = "malvar_he_cutler",
        enable_lens_shading: bool = True,
        lens_shading_poly: List[float] = None,
        enable_dpc: bool = True,
        dpc_threshold: float = 0.25,
        wb_gains: Optional[List[float]] = None,
        ccm: Optional[List[float]] = None,
        enable_denoise: bool = True,
        denoise_radius: int = 2,
        denoise_sigma_s: float = 2.0,
        denoise_sigma_r: float = 0.1,
        enable_sharpen: bool = True,
        sharpen_strength: float = 0.4,
        sharpen_threshold: float = 0.02,
        tonemap_operator: str = "aces"
    ):
        self.demosaic_algorithm = demosaic_algorithm
        self.enable_lens_shading = enable_lens_shading
        self.lens_shading_poly = lens_shading_poly or [1.0, 0.05, 0.01, 0.0]
        self.enable_dpc = enable_dpc
        self.dpc_threshold = dpc_threshold
        self.wb_gains = wb_gains or [2.0, 1.0, 1.8]
        self.ccm = ccm or DEFAULT_CCM_SRGB
        self.enable_denoise = enable_denoise
        self.denoise_radius = denoise_radius
        self.denoise_sigma_s = denoise_sigma_s
        self.denoise_sigma_r = denoise_sigma_r
        self.enable_sharpen = enable_sharpen
        self.sharpen_strength = sharpen_strength
        self.sharpen_threshold = sharpen_threshold
        self.tonemap_operator = tonemap_operator

class ISPPipeline:
    """Full-featured Image Signal Processor pipeline."""

    def __init__(self, config: Optional[ISPPipelineConfig] = None):
        self.config = config or ISPPipelineConfig()

    def process(self, raw: RawBayerImage) -> List[float]:
        """Processes a raw single-channel Bayer frame into high-fidelity RGB."""
        w, h = raw.width, raw.height

        # Stage 1: Black level subtraction and normalization
        linear = apply_black_level_subtraction(raw, raw.black_level, raw.white_level)

        # Stage 2: Lens shading correction
        if self.config.enable_lens_shading:
            linear = apply_lens_shading_correction(linear, w, h, self.config.lens_shading_poly)

        # Stage 3: Defective pixel correction
        if self.config.enable_dpc:
            linear = apply_defective_pixel_correction(linear, w, h, self.config.dpc_threshold)

        # Stage 4: CFA Demosaicing
        algo = self.config.demosaic_algorithm
        if algo == "malvar_he_cutler":
            rgb = demosaic_malvar_he_cutler(linear, w, h, raw.cfa_pattern)
        elif algo == "directional":
            rgb = demosaic_directional(linear, w, h, raw.cfa_pattern)
        else:
            rgb = demosaic_bilinear(linear, w, h, raw.cfa_pattern)

        # Stage 5: White Balance
        gr, gg, gb = self.config.wb_gains
        rgb = apply_white_balance(rgb, gr, gg, gb)

        # Stage 6: Color Correction Matrix
        rgb = apply_color_correction_matrix(rgb, self.config.ccm)

        # Stage 7: Denoising
        if self.config.enable_denoise and w >= 8 and h >= 8:
            rgb = apply_bilateral_filter(
                rgb, w, h,
                channels=3,
                radius=self.config.denoise_radius,
                sigma_spatial=self.config.denoise_sigma_s,
                sigma_range=self.config.denoise_sigma_r
            )

        # Stage 8: Detail Sharpening
        if self.config.enable_sharpen:
            rgb = apply_unsharp_mask(
                rgb, w, h,
                channels=3,
                strength=self.config.sharpen_strength,
                threshold=self.config.sharpen_threshold
            )

        # Stage 9: Tone Mapping & Transfer Function
        if self.config.tonemap_operator == "aces":
            rgb = tone_map_aces(rgb)
        elif self.config.tonemap_operator == "reinhard":
            rgb = tone_map_reinhard(rgb)

        rgb = apply_srgb_eotf(rgb)
        return rgb
