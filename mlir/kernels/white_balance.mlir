// Chromatic Adaptation and White Balance Kernel in MLIR
// Scales RGB channels independently using von Kries diagonal transformation.

module @white_balance {
  func.func @white_balance_3d(
      %input: memref<?x?x3xf32>,
      %output: memref<?x?x3xf32>,
      %gain_r: f32,
      %gain_g: f32,
      %gain_b: f32) {
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
        %r_in = memref.load %input[%y, %x, %ch_r] : memref<?x?x3xf32>
        %g_in = memref.load %input[%y, %x, %ch_g] : memref<?x?x3xf32>
        %b_in = memref.load %input[%y, %x, %ch_b] : memref<?x?x3xf32>

        %r_scaled = arith.mulf %r_in, %gain_r : f32
        %g_scaled = arith.mulf %g_in, %gain_g : f32
        %b_scaled = arith.mulf %b_in, %gain_b : f32

        %r_clamp1 = arith.maximumf %r_scaled, %f0 : f32
        %r_out = arith.minimumf %r_clamp1, %f1 : f32

        %g_clamp1 = arith.maximumf %g_scaled, %f0 : f32
        %g_out = arith.minimumf %g_clamp1, %f1 : f32

        %b_clamp1 = arith.maximumf %b_scaled, %f0 : f32
        %b_out = arith.minimumf %b_clamp1, %f1 : f32

        memref.store %r_out, %output[%y, %x, %ch_r] : memref<?x?x3xf32>
        memref.store %g_out, %output[%y, %x, %ch_g] : memref<?x?x3xf32>
        memref.store %b_out, %output[%y, %x, %ch_b] : memref<?x?x3xf32>
      }
    }
    return
  }
}
