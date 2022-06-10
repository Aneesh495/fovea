# Fovea Dialect Specification

This document provides the formal specification of the `fovea` domain-specific MLIR dialect. The dialect is formally defined in TableGen (`dialects/tablegen/FoveaOps.td`) and verified dynamically in MLIR 23 IRDL (`dialects/irdl/fovea.irdl.mlir`).

## 1. Dialect Overview

- **Dialect Name**: `fovea`
- **Dialect Namespace**: `fovea`
- **Summary**: Domain-specific intermediate representation for raw sensor processing, Bayer demosaicing, and high-throughput computational photography.
- **Header Files**: `dialects/tablegen/FoveaDialect.td`, `dialects/tablegen/FoveaOps.td`, `dialects/irdl/fovea.irdl.mlir`

---

## 2. Operations Specification

### `fovea.black_level_sub`
Subtracts optical black pedestal baseline values per Bayer color channel and rescales dynamic range to unit interval $[0.0, 1.0]$.

- **Syntax**: `%out = fovea.black_level_sub(%raw, %pedestal) : (memref<?x?xf32>, f32) -> memref<?x?xf32>`
- **Traits**: `NoMemoryEffect`, `SameOperandsAndResultShape`
- **Operands**:
  - `%raw`: 2D single-channel Bayer mosaic memref (`memref<?x?xf32>`)
  - `%pedestal`: Sensor optical black level offset (`f32`)
- **Results**:
  - `%out`: Calibrated Bayer mosaic memref (`memref<?x?xf32>`)

---

### `fovea.lens_shading_correction`
Compensates for cosine-fourth optical falloff and lens vignetting using a radial polynomial expansion centered at optical principal point $(c_x, c_y)$.

- **Syntax**: `%out = fovea.lens_shading_correction(%in, %poly_coeffs) : (memref<?x?xf32>, tensor<4xf32>) -> memref<?x?xf32>`
- **Traits**: `SameOperandsAndResultShape`
- **Operands**:
  - `%in`: Calibrated input mosaic (`memref<?x?xf32>`)
  - `%poly_coeffs`: Radial polynomial coefficients $[k_0, k_1, k_2, k_3]$ where gain $G(r) = 1 + k_0 r^2 + k_1 r^4 + k_2 r^6 + k_3 r^8$ (`tensor<4xf32>`)
- **Results**:
  - `%out`: Vignette-corrected Bayer mosaic (`memref<?x?xf32>`)

---

### `fovea.defective_pixel_correction`
Dynamic defective pixel detection and correction. Identifies hot or dead sensor pixels by testing whether the target sample deviates from the median of its 8-connected neighborhood by more than a calibrated variance threshold.

- **Syntax**: `%out = fovea.defective_pixel_correction(%in, %threshold) : (memref<?x?xf32>, f32) -> memref<?x?xf32>`
- **Traits**: `SameOperandsAndResultShape`
- **Operands**:
  - `%in`: Raw Bayer mosaic frame (`memref<?x?xf32>`)
  - `%threshold`: Gradient deviation rejection threshold (`f32`)
- **Results**:
  - `%out`: Outlier-free Bayer mosaic frame (`memref<?x?xf32>`)

---

### `fovea.bayer_demosaic`
Reconstructs full three-channel RGB color imagery from single-channel Bayer mosaic sampling.

- **Syntax**: `%rgb = fovea.bayer_demosaic(%mosaic) { pattern = "RGGB", method = "malvar_he_cutler" } : (memref<?x?xf32>) -> memref<?x?x3xf32>`
- **Attributes**:
  - `pattern`: String attribute defining Bayer phase (`"RGGB"`, `"BGGR"`, `"GRBG"`, `"GBRG"`)
  - `method`: Demosaicing algorithm (`"bilinear"`, `"malvar_he_cutler"`, `"directional"`)
- **Operands**:
  - `%mosaic`: 2D single-channel Bayer mosaic (`memref<?x?xf32>`)
- **Results**:
  - `%rgb`: 3D multi-channel RGB image (`memref<?x?x3xf32>`)

---

### `fovea.white_balance`
Applies von Kries diagonal illuminant adaptation factors per color channel.

- **Syntax**: `%out = fovea.white_balance(%rgb, %gains) : (memref<?x?x3xf32>, tensor<3xf32>) -> memref<?x?x3xf32>`
- **Operands**:
  - `%rgb`: Multi-channel RGB image (`memref<?x?x3xf32>`)
  - `%gains`: Diagonal channel gains $[g_r, g_g, g_b]$ (`tensor<3xf32>`)
- **Results**:
  - `%out`: White-balanced image (`memref<?x?x3xf32>`)

---

### `fovea.color_correction_matrix`
Transforms camera-native sensor spectral sensitivities into linear CIE XYZ or standard sRGB coordinates via a calibrated 3x3 matrix multiplication.

- **Syntax**: `%out = fovea.color_correction_matrix(%rgb, %matrix) : (memref<?x?x3xf32>, tensor<9xf32>) -> memref<?x?x3xf32>`
- **Operands**:
  - `%rgb`: Linear RGB image (`memref<?x?x3xf32>`)
  - `%matrix`: Flattened 3x3 transformation matrix (`tensor<9xf32>`)
- **Results**:
  - `%out`: Color-calibrated RGB image (`memref<?x?x3xf32>`)

---

### `fovea.joint_bilateral`
Non-linear edge-preserving filter combining a spatial Gaussian kernel with an intensity range Gaussian kernel.

- **Syntax**: `%out = fovea.joint_bilateral(%rgb, %guide, %sigma_s, %sigma_r) : (memref<?x?x3xf32>, memref<?x?xf32>, f32, f32) -> memref<?x?x3xf32>`
- **Operands**:
  - `%rgb`: Input noisy RGB image (`memref<?x?x3xf32>`)
  - `%guide`: High-SNR guidance image, typically green channel or luminance (`memref<?x?xf32>`)
  - `%sigma_s`: Spatial distance standard deviation (`f32`)
  - `%sigma_r`: Photometric intensity range standard deviation (`f32`)
- **Results**:
  - `%out`: Denoised edge-preserved image (`memref<?x?x3xf32>`)

---

### `fovea.guided_filter`
O(1) time-complexity edge-preserving filter based on a local linear translation model. Avoids gradient reversal artifacts common to bilateral filtering.

- **Syntax**: `%out = fovea.guided_filter(%rgb, %guide, %radius, %eps) : (memref<?x?x3xf32>, memref<?x?xf32>, i32, f32) -> memref<?x?x3xf32>`
- **Operands**:
  - `%rgb`: Input image (`memref<?x?x3xf32>`)
  - `%guide`: Structural guidance image (`memref<?x?xf32>`)
  - `%radius`: Moving window box radius (`i32`)
  - `%eps`: Regularization parameter preventing division by zero (`f32`)
- **Results**:
  - `%out`: Filtered image (`memref<?x?x3xf32>`)

---

### `fovea.unsharp_mask`
High-frequency edge enhancement filter using a separable binomial blur with coring threshold suppression.

- **Syntax**: `%out = fovea.unsharp_mask(%rgb, %strength, %threshold) : (memref<?x?x3xf32>, f32, f32) -> memref<?x?x3xf32>`
- **Operands**:
  - `%rgb`: Input RGB image (`memref<?x?x3xf32>`)
  - `%strength`: High-frequency amplification gain (`f32`)
  - `%threshold`: Coring threshold below which subtle noise is ignored (`f32`)
- **Results**:
  - `%out`: Sharpened image (`memref<?x?x3xf32>`)

---

### `fovea.pyramid_downsample`
Dyadic spatial downsampling using a 5-tap separable binomial low-pass filter $[1, 4, 6, 4, 1] / 16$.

- **Syntax**: `%down = fovea.pyramid_downsample(%in) : (memref<?x?x3xf32>) -> memref<?x?x3xf32>`
- **Results**:
  - `%down`: Reduced image with dimensions $(\lfloor W/2 \rfloor, \lfloor H/2 \rfloor)$ (`memref<?x?x3xf32>`)

---

### `fovea.pyramid_upsample`
Dyadic spatial upsampling with bilinear zero-insertion and binomial interpolation.

- **Syntax**: `%up = fovea.pyramid_upsample(%in, %target_w, %target_h) : (memref<?x?x3xf32>, i32, i32) -> memref<?x?x3xf32>`
- **Results**:
  - `%up`: Expanded image matching specified target dimensions (`memref<?x?x3xf32>`)

---

### `fovea.mertens_hdr_fusion`
Multi-exposure burst fusion combining contrast, saturation, and well-exposedness quality metrics into seamless composite dynamic range.

- **Syntax**: `%fused = fovea.mertens_hdr_fusion(%bracket, %weights) : (memref<?x?x?x3xf32>, tensor<3xf32>) -> memref<?x?x3xf32>`
- **Operands**:
  - `%bracket`: 4D sequence of bracketed exposure frames (`memref<?x?x?x3xf32>`)
  - `%weights`: Relative metric exponents $[w_{\text{contrast}}, w_{\text{saturation}}, w_{\text{exposedness}}]$ (`tensor<3xf32>`)
- **Results**:
  - `%fused`: High dynamic range composite frame (`memref<?x?x3xf32>`)

---

### `fovea.reinhard_tone_map`
Global photographic key tone reproduction mapping unbounded scene radiance into $[0.0, 1.0]$.

- **Syntax**: `%out = fovea.reinhard_tone_map(%in, %key, %white_point) : (memref<?x?x3xf32>, f32, f32) -> memref<?x?x3xf32>`
- **Operands**:
  - `%in`: Unbounded high dynamic range image (`memref<?x?x3xf32>`)
  - `%key`: Photographic exposure key parameter $\alpha$ (`f32`)
  - `%white_point`: Minimum luminance mapped to pure white $L_{\text{white}}$ (`f32`)
- **Results**:
  - `%out`: Tone-mapped display image (`memref<?x?x3xf32>`)

---

### `fovea.aces_tone_map`
Closed-form rational polynomial approximation of the Academy Color Encoding System (ACES) filmic tone reproduction curve.

- **Syntax**: `%out = fovea.aces_tone_map(%in) : (memref<?x?x3xf32>) -> memref<?x?x3xf32>`
- **Formula**:
  $$
  f(x) = \frac{x(2.51x + 0.03)}{x(2.43x + 0.59) + 0.14}
  $$
- **Operands**:
  - `%in`: High dynamic range image (`memref<?x?x3xf32>`)
- **Results**:
  - `%out`: Filmic compressed display image (`memref<?x?x3xf32>`)
