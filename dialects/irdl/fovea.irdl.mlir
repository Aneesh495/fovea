// Fovea Dialect IRDL Definition
// High-level intermediate representation for camera image signal processing (ISP)
// and computational photography pipelines.

module {
  irdl.dialect @fovea {

    // Attribute: Bayer Color Filter Array (CFA) Pattern
    irdl.type @cfa_layout

    // Operation: Black Level Subtraction and Pedestal Compensation
    // Linearizes raw analog-to-digital converter (ADC) pixel intensities.
    irdl.operation @black_level {
      %elem = irdl.any
      irdl.operands(raw_in: %elem)
      irdl.results(linearized_out: %elem)
    }

    // Operation: Radial Lens Shading and Optical Vignetting Correction
    // Multiplies radial distance polynomial to correct optical attenuation.
    irdl.operation @lens_shading {
      %elem = irdl.any
      irdl.operands(raw_in: %elem)
      irdl.results(corrected_out: %elem)
    }

    // Operation: Defective Pixel Detection and Dynamic Median Replacement
    // Identifies stuck hot and dead pixels using cross-neighborhood thresholding.
    irdl.operation @defective_pixel {
      %elem = irdl.any
      irdl.operands(raw_in: %elem)
      irdl.results(filtered_out: %elem)
    }

    // Operation: Color Filter Array (CFA) Demosaicing
    // Interpolates single-channel mosaic into full-resolution multi-channel RGB.
    // Supports Bilinear, Directional Gradient, and Malvar-He-Cutler optimal FIR.
    irdl.operation @demosaic {
      %raw = irdl.any
      %rgb = irdl.any
      irdl.operands(raw_in: %raw)
      irdl.results(rgb_out: %rgb)
    }

    // Operation: Chromatic Adaptation and White Balance
    // Balances spectral sensitivity using von Kries or Bradford diagonal scaling.
    irdl.operation @white_balance {
      %rgb = irdl.any
      irdl.operands(rgb_in: %rgb)
      irdl.results(wb_out: %rgb)
    }

    // Operation: 3x3 Sensor Color Correction Matrix (CCM)
    // Projects sensor spectral response to standard color spaces (sRGB, Rec.709).
    irdl.operation @color_correction {
      %rgb = irdl.any
      irdl.operands(rgb_in: %rgb)
      irdl.results(ccm_out: %rgb)
    }

    // Operation: Edge-Preserving Joint Bilateral Filtering
    // Combines spatial Gaussian proximity and radiometric photometric distance.
    irdl.operation @bilateral {
      %rgb = irdl.any
      irdl.operands(rgb_in: %rgb)
      irdl.results(denoised_out: %rgb)
    }

    // Operation: O(1) Fast Guided Image Filter
    // Linear local model transfer for gradient-preserving detail enhancement.
    irdl.operation @guided_filter {
      %guide = irdl.any
      %input = irdl.any
      %out = irdl.any
      irdl.operands(guide_in: %guide, input_in: %input)
      irdl.results(filtered_out: %out)
    }

    // Operation: Multi-Scale Unsharp Masking
    // Enhances high-frequency spatial gradients with coring and halo suppression.
    irdl.operation @unsharp_mask {
      %rgb = irdl.any
      irdl.operands(rgb_in: %rgb)
      irdl.results(sharpened_out: %rgb)
    }

    // Operation: Mertens-Kautz-Van Reeth Multi-Exposure HDR Fusion
    // Seamlessly blends exposure brackets using local contrast, saturation, and well-exposedness.
    irdl.operation @hdr_fuse {
      %bracket = irdl.any
      %hdr = irdl.any
      irdl.operands(under_exp: %bracket, mid_exp: %bracket, over_exp: %bracket)
      irdl.results(fused_out: %hdr)
    }

    // Operation: 5-Tap Separable Binomial Gaussian Pyramid Downsampler
    // Multi-scale dyadic spatial decomposition.
    irdl.operation @pyr_down {
      %in = irdl.any
      %out = irdl.any
      irdl.operands(input: %in)
      irdl.results(downsampled: %out)
    }

    // Operation: Dyadic Pyramid Upsampler and Interpolator
    // Dual expansion operator for Laplacian pyramid reconstruction.
    irdl.operation @pyr_up {
      %in = irdl.any
      %out = irdl.any
      irdl.operands(input: %in)
      irdl.results(upsampled: %out)
    }

    // Operation: Perceptual Dynamic Range Compression and Tone Mapping
    // Maps wide dynamic range radiance to displayable range (Reinhard, Drago, ACES).
    irdl.operation @tone_map {
      %rgb = irdl.any
      irdl.operands(hdr_in: %rgb)
      irdl.results(ldr_out: %rgb)
    }

    // Operation: Electro-Optical Transfer Function (EOTF) and Gamma Curve
    // Standard IEC 61966-2-1 sRGB or Rec.709 power curve transformation.
    irdl.operation @gamma {
      %rgb = irdl.any
      irdl.operands(rgb_in: %rgb)
      irdl.results(gamma_out: %rgb)
    }
  }
}
