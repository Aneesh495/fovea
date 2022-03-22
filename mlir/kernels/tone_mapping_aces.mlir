// ACES Filmic Tone Reproduction Curve Kernel in MLIR
// Evaluates rational polynomial fit: f(x) = (x * (a*x + b)) / (x * (c*x + d) + e)
// where a=2.51, b=0.03, c=2.43, d=0.59, e=0.14.

module @tone_mapping_aces {
  func.func @tone_map_aces_3d(
      %hdr_in: memref<?x?x3xf32>,
      %ldr_out: memref<?x?x3xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index

    %f0 = arith.constant 0.000000e+00 : f32
    %f1 = arith.constant 1.000000e+00 : f32

    %a = arith.constant 2.510000e+00 : f32
    %b = arith.constant 0.030000e+00 : f32
    %c = arith.constant 2.430000e+00 : f32
    %d = arith.constant 0.590000e+00 : f32
    %e = arith.constant 0.140000e+00 : f32

    %height = memref.dim %hdr_in, %c0 : memref<?x?x3xf32>
    %width = memref.dim %hdr_in, %c1 : memref<?x?x3xf32>

    scf.for %y = %c0 to %height step %c1 {
      scf.for %x = %c0 to %width step %c1 {
        scf.for %ch = %c0 to %c3 step %c1 {
          %val = memref.load %hdr_in[%y, %x, %ch] : memref<?x?x3xf32>
          %x_pos = arith.maximumf %val, %f0 : f32

          // Numerator: x * (a * x + b)
          %ax = arith.mulf %a, %x_pos : f32
          %ax_b = arith.addf %ax, %b : f32
          %num = arith.mulf %x_pos, %ax_b : f32

          // Denominator: x * (c * x + d) + e
          %cx = arith.mulf %c, %x_pos : f32
          %cx_d = arith.addf %cx, %d : f32
          %denom_term = arith.mulf %x_pos, %cx_d : f32
          %denom = arith.addf %denom_term, %e : f32

          %mapped = arith.divf %num, %denom : f32
          %c1_val = arith.maximumf %mapped, %f0 : f32
          %out = arith.minimumf %c1_val, %f1 : f32

          memref.store %out, %ldr_out[%y, %x, %ch] : memref<?x?x3xf32>
        }
      }
    }
    return
  }
}
