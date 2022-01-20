// Color Correction Matrix (CCM) Kernel in MLIR
// Applies a 3x3 linear transformation to convert sensor spectral responses
// to target color spaces (sRGB, Rec.709, DCI-P3) with gamut clamping.

module @color_correction_matrix {
  func.func @color_correction_3d(
      %input: memref<?x?x3xf32>,
      %output: memref<?x?x3xf32>,
      %m00: f32, %m01: f32, %m02: f32,
      %m10: f32, %m11: f32, %m12: f32,
      %m20: f32, %m21: f32, %m22: f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ch_r = arith.constant 0 : index
    %ch_g = arith.constant 1 : index
    %ch_b = arith.constant 2 : index

    %f0 = arith.constant 0.000000e+00 : f32
    %f1 = arith.constant 1.000000e+00 : f32

    %height = memref.dim %input, %c0 : memref<?x?x3xf32>
    %width = memref.dim %input, %c1 : memref<?x?x3xf32>

    scf.for %y = %c0 to %height step %c1 {
      scf.for %x = %c0 to %width step %c1 {
        %r = memref.load %input[%y, %x, %ch_r] : memref<?x?x3xf32>
        %g = memref.load %input[%y, %x, %ch_g] : memref<?x?x3xf32>
        %b = memref.load %input[%y, %x, %ch_b] : memref<?x?x3xf32>

        // R_out = m00*r + m01*g + m02*b
        %r0 = arith.mulf %m00, %r : f32
        %r1 = arith.mulf %m01, %g : f32
        %r2 = arith.mulf %m02, %b : f32
        %r_sum1 = arith.addf %r0, %r1 : f32
        %r_out = arith.addf %r_sum1, %r2 : f32

        // G_out = m10*r + m11*g + m12*b
        %g0 = arith.mulf %m10, %r : f32
        %g1 = arith.mulf %m11, %g : f32
        %g2 = arith.mulf %m12, %b : f32
        %g_sum1 = arith.addf %g0, %g1 : f32
        %g_out = arith.addf %g_sum1, %g2 : f32

        // B_out = m20*r + m21*g + m22*b
        %b0 = arith.mulf %m20, %r : f32
        %b1 = arith.mulf %m21, %g : f32
        %b2 = arith.mulf %m22, %b : f32
        %b_sum1 = arith.addf %b0, %b1 : f32
        %b_out = arith.addf %b_sum1, %b2 : f32

        // Clamping to [0.0, 1.0]
        %r_clamped1 = arith.maximumf %r_out, %f0 : f32
        %r_clamped = arith.minimumf %r_clamped1, %f1 : f32

        %g_clamped1 = arith.maximumf %g_out, %f0 : f32
        %g_clamped = arith.minimumf %g_clamped1, %f1 : f32

        %b_clamped1 = arith.maximumf %b_out, %f0 : f32
        %b_clamped = arith.minimumf %b_clamped1, %f1 : f32

        memref.store %r_clamped, %output[%y, %x, %ch_r] : memref<?x?x3xf32>
        memref.store %g_clamped, %output[%y, %x, %ch_g] : memref<?x?x3xf32>
        memref.store %b_clamped, %output[%y, %x, %ch_b] : memref<?x?x3xf32>
      }
    }
    return
  }
}
