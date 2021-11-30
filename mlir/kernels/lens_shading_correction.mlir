// Radial Lens Shading and Vignetting Correction Kernel in MLIR
// Evaluates a radial 6th-order polynomial gain map to compensate for
// optical cos^4(theta) falloff and mechanical aperture vignetting.

module @lens_shading_correction {
  func.func @lens_shading_2d(
      %input: memref<?x?xf32>,
      %output: memref<?x?xf32>,
      %cx: f32,
      %cy: f32,
      %inv_max_r: f32,
      %k0: f32,
      %k1: f32,
      %k2: f32,
      %k3: f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %f0 = arith.constant 0.000000e+00 : f32
    %f1 = arith.constant 1.000000e+00 : f32

    %dim0 = memref.dim %input, %c0 : memref<?x?xf32>
    %dim1 = memref.dim %input, %c1 : memref<?x?xf32>

    scf.for %i = %c0 to %dim0 step %c1 {
      %i_int = arith.index_cast %i : index to i32
      %i_f = arith.sitofp %i_int : i32 to f32
      %dy = arith.subf %i_f, %cy : f32
      %dy2 = arith.mulf %dy, %dy : f32

      scf.for %j = %c0 to %dim1 step %c1 {
        %j_int = arith.index_cast %j : index to i32
        %j_f = arith.sitofp %j_int : i32 to f32
        %dx = arith.subf %j_f, %cx : f32
        %dx2 = arith.mulf %dx, %dx : f32
        %r2_unscaled = arith.addf %dx2, %dy2 : f32
        %r_unscaled = math.sqrt %r2_unscaled : f32
        %r_norm = arith.mulf %r_unscaled, %inv_max_r : f32

        %r2 = arith.mulf %r_norm, %r_norm : f32
        %r4 = arith.mulf %r2, %r2 : f32
        %r6 = arith.mulf %r4, %r2 : f32

        %term1 = arith.mulf %k1, %r2 : f32
        %term2 = arith.mulf %k2, %r4 : f32
        %term3 = arith.mulf %k3, %r6 : f32

        %g01 = arith.addf %k0, %term1 : f32
        %g012 = arith.addf %g01, %term2 : f32
        %gain = arith.addf %g012, %term3 : f32

        %val = memref.load %input[%i, %j] : memref<?x?xf32>
        %corrected = arith.mulf %val, %gain : f32
        %clamped_min = arith.maximumf %corrected, %f0 : f32
        %clamped = arith.minimumf %clamped_min, %f1 : f32
        memref.store %clamped, %output[%i, %j] : memref<?x?xf32>
      }
    }
    return
  }
}
