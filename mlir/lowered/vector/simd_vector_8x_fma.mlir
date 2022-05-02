// SIMD Vector Kernel: Vector Width 8, Operation fma
module @simd_fma_v8 {
  func.func @execute_simd_fma(
      %src1: memref<16384xf32>,
      %src2: memref<16384xf32>,
      %dst: memref<16384xf32>,
      %factor: f32) {
    %c0 = arith.constant 0 : index
    %total = arith.constant 16384 : index
    %step = arith.constant 8 : index
    %f0 = arith.constant 0.000000e+00 : f32
    %v_factor = vector.broadcast %factor : f32 to vector<8xf32>

    scf.for %i = %c0 to %total step %step {
      %v1 = vector.transfer_read %src1[%i], %f0 : memref<16384xf32>, vector<8xf32>
      %v2 = vector.transfer_read %src2[%i], %f0 : memref<16384xf32>, vector<8xf32>
      %mul = arith.mulf %v1, %v_factor : vector<8xf32>
      %res = arith.addf %mul, %v2 : vector<8xf32>
      vector.transfer_write %res, %dst[%i] : vector<8xf32>, memref<16384xf32>
    }
    return
  }
}
