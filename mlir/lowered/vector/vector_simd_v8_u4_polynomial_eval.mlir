// Horner Polynomial Evaluation - SIMD Vector Width 8, Loop Unroll Factor 4x
module @simd_v8_u4_polynomial_eval {
  func.func @execute_simd_unroll(
      %in: memref<32768xf32>,
      %out: memref<32768xf32>,
      %scale: f32) {
    %c0 = arith.constant 0 : index
    %total = arith.constant 32768 : index
    %step = arith.constant 32 : index
    %f0 = arith.constant 0.000000e+00 : f32
    %v_scale = vector.broadcast %scale : f32 to vector<8xf32>

    scf.for %i = %c0 to %total step %step {
      %c_off0 = arith.constant 0 : index
      %idx0 = arith.addi %i, %c_off0 : index
      %v_in0 = vector.transfer_read %in[%idx0], %f0 : memref<32768xf32>, vector<8xf32>
      %v_res0 = arith.mulf %v_in0, %v_scale : vector<8xf32>
      vector.transfer_write %v_res0, %out[%idx0] : vector<8xf32>, memref<32768xf32>
      %c_off1 = arith.constant 8 : index
      %idx1 = arith.addi %i, %c_off1 : index
      %v_in1 = vector.transfer_read %in[%idx1], %f0 : memref<32768xf32>, vector<8xf32>
      %v_res1 = arith.mulf %v_in1, %v_scale : vector<8xf32>
      vector.transfer_write %v_res1, %out[%idx1] : vector<8xf32>, memref<32768xf32>
      %c_off2 = arith.constant 16 : index
      %idx2 = arith.addi %i, %c_off2 : index
      %v_in2 = vector.transfer_read %in[%idx2], %f0 : memref<32768xf32>, vector<8xf32>
      %v_res2 = arith.mulf %v_in2, %v_scale : vector<8xf32>
      vector.transfer_write %v_res2, %out[%idx2] : vector<8xf32>, memref<32768xf32>
      %c_off3 = arith.constant 24 : index
      %idx3 = arith.addi %i, %c_off3 : index
      %v_in3 = vector.transfer_read %in[%idx3], %f0 : memref<32768xf32>, vector<8xf32>
      %v_res3 = arith.mulf %v_in3, %v_scale : vector<8xf32>
      vector.transfer_write %v_res3, %out[%idx3] : vector<8xf32>, memref<32768xf32>
    }
    return
  }
}
