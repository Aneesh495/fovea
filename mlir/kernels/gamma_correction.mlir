// Gamma Correction and Electro-Optical Transfer Function Kernel in MLIR
// Applies sRGB power curve: V_out = V_in^(1.0 / gamma).

module @gamma_correction {
  func.func @gamma_3d(
      %input: memref<?x?x3xf32>,
      %output: memref<?x?x3xf32>,
      %inv_gamma: f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %f0 = arith.constant 0.000000e+00 : f32
    %f1 = arith.constant 1.000000e+00 : f32

    %height = memref.dim %input, %c0 : memref<?x?x3xf32>
    %width = memref.dim %input, %c1 : memref<?x?x3xf32>

    scf.for %y = %c0 to %height step %c1 {
      scf.for %x = %c0 to %width step %c1 {
        scf.for %ch = %c0 to %c3 step %c1 {
          %val = memref.load %input[%y, %x, %ch] : memref<?x?x3xf32>
          %clamped1 = arith.maximumf %val, %f0 : f32
          %clamped = arith.minimumf %clamped1, %f1 : f32

          // pow(c, inv_gamma) = exp(inv_gamma * log(c))
          // For positive c, use math.powf
          %res = math.powf %clamped, %inv_gamma : f32
          %res_c1 = arith.maximumf %res, %f0 : f32
          %out = arith.minimumf %res_c1, %f1 : f32

          memref.store %out, %output[%y, %x, %ch] : memref<?x?x3xf32>
        }
      }
    }
    return
  }
}
