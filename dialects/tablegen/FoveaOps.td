// Fovea Operations TableGen Definition

#ifndef FOVEA_OPS_TD
#define FOVEA_OPS_TD

include "FoveaDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

class Fovea_Op<string mnemonic, list<Trait> traits = []> :
    Op<Fovea_Dialect, mnemonic, traits>;

// Black Level Subtraction
def Fovea_BlackLevelOp : Fovea_Op<"black_level", [Pure]> {
  let summary = "Linearizes raw sensor readings by subtracting ADC pedestal offset";
  let arguments = (ins
    AnyTensor:$input,
    F32Attr:$black_level_offset,
    F32Attr:$saturation_limit
  );
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

// Lens Shading Correction
def Fovea_LensShadingOp : Fovea_Op<"lens_shading", [Pure]> {
  let summary = "Corrects radial optical vignetting using a 4th-order polynomial";
  let arguments = (ins
    AnyTensor:$input,
    F32ArrayAttr:$radial_poly_coefficients,
    F32ArrayAttr:$optical_center
  );
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

// Defective Pixel Correction
def Fovea_DefectivePixelOp : Fovea_Op<"defective_pixel", [Pure]> {
  let summary = "Detects and replaces stuck hot and dead pixels using neighborhood median";
  let arguments = (ins
    AnyTensor:$input,
    F32Attr:$variance_threshold
  );
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

// CFA Demosaicing
def Fovea_DemosaicOp : Fovea_Op<"demosaic", [Pure]> {
  let summary = "Interpolates single-channel CFA Bayer mosaic to full 3-channel RGB";
  let arguments = (ins
    AnyTensor:$input,
    StrAttr:$cfa_pattern,
    StrAttr:$algorithm
  );
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

// White Balance
def Fovea_WhiteBalanceOp : Fovea_Op<"white_balance", [Pure]> {
  let summary = "Applies diagonal chromatic adaptation gains to RGB channels";
  let arguments = (ins
    AnyTensor:$input,
    F32Attr:$gain_r,
    F32Attr:$gain_g,
    F32Attr:$gain_b
  );
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

// Color Correction Matrix (CCM)
def Fovea_ColorCorrectionOp : Fovea_Op<"color_correction", [Pure]> {
  let summary = "Applies 3x3 spectral color correction matrix";
  let arguments = (ins
    AnyTensor:$input,
    F32ArrayAttr:$matrix_3x3
  );
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

// Joint Bilateral Filter
def Fovea_BilateralOp : Fovea_Op<"bilateral", [Pure]> {
  let summary = "Edge-preserving spatial and photometric smoothing filter";
  let arguments = (ins
    AnyTensor:$input,
    I32Attr:$radius,
    F32Attr:$sigma_spatial,
    F32Attr:$sigma_range
  );
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

// Guided Filter
def Fovea_GuidedFilterOp : Fovea_Op<"guided_filter", [Pure]> {
  let summary = "Fast edge-preserving linear local model guided filter";
  let arguments = (ins
    AnyTensor:$guide,
    AnyTensor:$input,
    I32Attr:$radius,
    F32Attr:$eps
  );
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$guide `,` $input attr-dict `:` type($guide) `,` type($input) `->` type($output)";
}

// Unsharp Masking
def Fovea_UnsharpMaskOp : Fovea_Op<"unsharp_mask", [Pure]> {
  let summary = "High-pass spatial frequency sharpening with coring threshold";
  let arguments = (ins
    AnyTensor:$input,
    F32Attr:$strength,
    F32Attr:$threshold
  );
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

// Mertens HDR Exposure Fusion
def Fovea_HdrFuseOp : Fovea_Op<"hdr_fuse", [Pure]> {
  let summary = "Mertens-Kautz-Van Reeth multi-exposure fusion";
  let arguments = (ins
    AnyTensor:$under_exp,
    AnyTensor:$mid_exp,
    AnyTensor:$over_exp,
    F32Attr:$w_contrast,
    F32Attr:$w_saturation,
    F32Attr:$w_exposedness
  );
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$under_exp `,` $mid_exp `,` $over_exp attr-dict `:` type($under_exp) `,` type($mid_exp) `,` type($over_exp) `->` type($output)";
}

// Pyramid Downsampler
def Fovea_PyrDownOp : Fovea_Op<"pyr_down", [Pure]> {
  let summary = "Downsamples image by 2x using 5-tap separable binomial filter";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

// Pyramid Upsampler
def Fovea_PyrUpOp : Fovea_Op<"pyr_up", [Pure]> {
  let summary = "Upsamples image by 2x using 5-tap separable interpolation filter";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

// Tone Mapping Operator
def Fovea_ToneMapOp : Fovea_Op<"tone_map", [Pure]> {
  let summary = "Perceptual dynamic range compression (Reinhard, Drago, ACES)";
  let arguments = (ins
    AnyTensor:$input,
    StrAttr:$operator_type,
    F32Attr:$key_value
  );
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

// Gamma Correction
def Fovea_GammaOp : Fovea_Op<"gamma", [Pure]> {
  let summary = "Applies IEC 61966-2-1 standard sRGB transfer curve";
  let arguments = (ins
    AnyTensor:$input,
    F32Attr:$gamma_value
  );
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

#endif // FOVEA_OPS_TD
