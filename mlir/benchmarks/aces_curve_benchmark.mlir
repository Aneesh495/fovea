// High-Throughput Benchmark Module for aces_curve in MLIR
module @aces_curve_benchmark {
  func.func @benchmark_step(
      %in: memref<2160x3840x3xf32>,
      %out: memref<2160x3840x3xf32>,
      %param: f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %h = arith.constant 2160 : index
    %w = arith.constant 3840 : index

    scf.for %y = %c0 to %h step %c1 {
      scf.for %x = %c0 to %w step %c1 {
        scf.for %c = %c0 to %c1 step %c1 {
          %v = memref.load %in[%y, %x, %c] : memref<2160x3840x3xf32>
          %res = arith.mulf %v, %param : f32
          memref.store %res, %out[%y, %x, %c] : memref<2160x3840x3xf32>
        }
      }
    }
    return
  }
}
