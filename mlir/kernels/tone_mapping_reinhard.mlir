// Reinhard Global Tone Reproduction Kernel in MLIR
// Compresses wide dynamic range luminance: L_d = L_scaled / (1.0 + L_scaled)
// while preserving color chromaticity ratios.

module @tone_mapping_reinhard {
  func.func @tone_map_reinhard_3d(
      %hdr_in: memref<?x?x3xf32>,
      %ldr_out: memref<?x?x3xf32>,
      %key_scale: f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ch_r = arith.constant 0 : index
    %ch_g = arith.constant 1 : index
    %ch_b = arith.constant 2 : index

    %f0 = arith.constant 0.000000e+00 : f32
    %f1 = arith.constant 1.000000e+00 : f32
    %eps = arith.constant 1.000000e-06 : f32

    // Rec.709 luminance weights: 0.2126 R + 0.7152 G + 0.0722 B
    %w_r = arith.constant 0.212600e+00 : f32
    %w_g = arith.constant 0.715200e+00 : f32
    %w_b = arith.constant 0.072200e+00 : f32

    %height = memref.dim %hdr_in, %c0 : memref<?x?x3xf32>
    %width = memref.dim %hdr_in, %c1 : memref<?x?x3xf32>

    scf.for %y = %c0 to %height step %c1 {
      scf.for %x = %c0 to %width step %c1 {
        %r = memref.load %hdr_in[%y, %x, %ch_r] : memref<?x?x3xf32>
        %g = memref.load %hdr_in[%y, %x, %ch_g] : memref<?x?x3xf32>
        %b = memref.load %hdr_in[%y, %x, %ch_b] : memref<?x?x3xf32>

        // Compute luminance L
        %lr = arith.mulf %r, %w_r : f32
        %lg = arith.mulf %g, %w_g : f32
        %lb = arith.mulf %b, %w_b : f32
        %l1 = arith.addf %lr, %lg : f32
        %lum = arith.addf %l1, %lb : f32

        // L_scaled = lum * key_scale
        %l_scaled = arith.mulf %lum, %key_scale : f32

        // L_d = L_scaled / (1.0 + L_scaled)
        %denom = arith.addf %f1, %l_scaled : f32
        %l_d = arith.divf %l_scaled, %denom : f32

        // Scale factor: L_d / (lum + eps)
        %lum_eps = arith.addf %lum, %eps : f32
        %scale = arith.divf %l_d, %lum_eps : f32

        %r_out_raw = arith.mulf %r, %scale : f32
        %g_out_raw = arith.mulf %g, %scale : f32
        %b_out_raw = arith.mulf %b, %scale : f32

        %r_c1 = arith.maximumf %r_out_raw, %f0 : f32
        %r_out = arith.minimumf %r_c1, %f1 : f32

        %g_c1 = arith.maximumf %g_out_raw, %f0 : f32
        %g_out = arith.minimumf %g_c1, %f1 : f32

        %b_c1 = arith.maximumf %b_out_raw, %f0 : f32
        %b_out = arith.minimumf %b_c1, %f1 : f32

        memref.store %r_out, %ldr_out[%y, %x, %ch_r] : memref<?x?x3xf32>
        memref.store %g_out, %ldr_out[%y, %x, %ch_g] : memref<?x?x3xf32>
        memref.store %b_out, %ldr_out[%y, %x, %ch_b] : memref<?x?x3xf32>
      }
    }
    return
  }
}
