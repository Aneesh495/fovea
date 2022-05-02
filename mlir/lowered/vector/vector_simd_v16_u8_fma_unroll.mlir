// Fused Multiply Add - SIMD Vector Width 16, Loop Unroll Factor 8x
module @simd_v16_u8_fma_unroll {
  func.func @execute_simd_unroll(
      %in: memref<32768xf32>,
      %out: memref<32768xf32>,
      %scale: f32) {
    %c0 = arith.constant 0 : index
    %total = arith.constant 32768 : index
    %step = arith.constant 128 : index
    %f0 = arith.constant 0.000000e+00 : f32
    %v_scale = vector.broadcast %scale : f32 to vector<16xf32>

    scf.for %i = %c0 to %total step %step {
      %c_off0 = arith.constant 0 : index
      %idx0 = arith.addi %i, %c_off0 : index
      %v_in0 = vector.transfer_read %in[%idx0], %f0 : memref<32768xf32>, vector<16xf32>
      %v_res0 = arith.mulf %v_in0, %v_scale : vector<16xf32>
      vector.transfer_write %v_res0, %out[%idx0] : vector<16xf32>, memref<32768xf32>
      %c_off1 = arith.constant 16 : index
      %idx1 = arith.addi %i, %c_off1 : index
      %v_in1 = vector.transfer_read %in[%idx1], %f0 : memref<32768xf32>, vector<16xf32>
      %v_res1 = arith.mulf %v_in1, %v_scale : vector<16xf32>
      vector.transfer_write %v_res1, %out[%idx1] : vector<16xf32>, memref<32768xf32>
      %c_off2 = arith.constant 32 : index
      %idx2 = arith.addi %i, %c_off2 : index
      %v_in2 = vector.transfer_read %in[%idx2], %f0 : memref<32768xf32>, vector<16xf32>
      %v_res2 = arith.mulf %v_in2, %v_scale : vector<16xf32>
      vector.transfer_write %v_res2, %out[%idx2] : vector<16xf32>, memref<32768xf32>
      %c_off3 = arith.constant 48 : index
      %idx3 = arith.addi %i, %c_off3 : index
      %v_in3 = vector.transfer_read %in[%idx3], %f0 : memref<32768xf32>, vector<16xf32>
      %v_res3 = arith.mulf %v_in3, %v_scale : vector<16xf32>
      vector.transfer_write %v_res3, %out[%idx3] : vector<16xf32>, memref<32768xf32>
      %c_off4 = arith.constant 64 : index
      %idx4 = arith.addi %i, %c_off4 : index
      %v_in4 = vector.transfer_read %in[%idx4], %f0 : memref<32768xf32>, vector<16xf32>
      %v_res4 = arith.mulf %v_in4, %v_scale : vector<16xf32>
      vector.transfer_write %v_res4, %out[%idx4] : vector<16xf32>, memref<32768xf32>
      %c_off5 = arith.constant 80 : index
      %idx5 = arith.addi %i, %c_off5 : index
      %v_in5 = vector.transfer_read %in[%idx5], %f0 : memref<32768xf32>, vector<16xf32>
      %v_res5 = arith.mulf %v_in5, %v_scale : vector<16xf32>
      vector.transfer_write %v_res5, %out[%idx5] : vector<16xf32>, memref<32768xf32>
      %c_off6 = arith.constant 96 : index
      %idx6 = arith.addi %i, %c_off6 : index
      %v_in6 = vector.transfer_read %in[%idx6], %f0 : memref<32768xf32>, vector<16xf32>
      %v_res6 = arith.mulf %v_in6, %v_scale : vector<16xf32>
      vector.transfer_write %v_res6, %out[%idx6] : vector<16xf32>, memref<32768xf32>
      %c_off7 = arith.constant 112 : index
      %idx7 = arith.addi %i, %c_off7 : index
      %v_in7 = vector.transfer_read %in[%idx7], %f0 : memref<32768xf32>, vector<16xf32>
      %v_res7 = arith.mulf %v_in7, %v_scale : vector<16xf32>
      vector.transfer_write %v_res7, %out[%idx7] : vector<16xf32>, memref<32768xf32>
    }
    return
  }
}
