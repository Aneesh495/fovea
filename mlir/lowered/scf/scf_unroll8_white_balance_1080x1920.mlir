// Unrolled SCF Loop Nest: Chromatic, Unroll 8x on 1080x1920
module @scf_unroll8_white_balance_1080x1920 {
  func.func @execute_unroll(
      %in: memref<1080x1920xf32>,
      %out: memref<1080x1920xf32>,
      %scale: f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %h_dim = arith.constant 1080 : index
    %w_dim = arith.constant 1920 : index
    %step = arith.constant 8 : index

    scf.for %y = %c0 to %h_dim step %c1 {
      scf.for %x = %c0 to %w_dim step %step {
        %c_off0 = arith.constant 0 : index
        %idx0 = arith.addi %x, %c_off0 : index
        %val0 = memref.load %in[%y, %idx0] : memref<1080x1920xf32>
        %res0 = arith.mulf %val0, %scale : f32
        memref.store %res0, %out[%y, %idx0] : memref<1080x1920xf32>
        %c_off1 = arith.constant 1 : index
        %idx1 = arith.addi %x, %c_off1 : index
        %val1 = memref.load %in[%y, %idx1] : memref<1080x1920xf32>
        %res1 = arith.mulf %val1, %scale : f32
        memref.store %res1, %out[%y, %idx1] : memref<1080x1920xf32>
        %c_off2 = arith.constant 2 : index
        %idx2 = arith.addi %x, %c_off2 : index
        %val2 = memref.load %in[%y, %idx2] : memref<1080x1920xf32>
        %res2 = arith.mulf %val2, %scale : f32
        memref.store %res2, %out[%y, %idx2] : memref<1080x1920xf32>
        %c_off3 = arith.constant 3 : index
        %idx3 = arith.addi %x, %c_off3 : index
        %val3 = memref.load %in[%y, %idx3] : memref<1080x1920xf32>
        %res3 = arith.mulf %val3, %scale : f32
        memref.store %res3, %out[%y, %idx3] : memref<1080x1920xf32>
        %c_off4 = arith.constant 4 : index
        %idx4 = arith.addi %x, %c_off4 : index
        %val4 = memref.load %in[%y, %idx4] : memref<1080x1920xf32>
        %res4 = arith.mulf %val4, %scale : f32
        memref.store %res4, %out[%y, %idx4] : memref<1080x1920xf32>
        %c_off5 = arith.constant 5 : index
        %idx5 = arith.addi %x, %c_off5 : index
        %val5 = memref.load %in[%y, %idx5] : memref<1080x1920xf32>
        %res5 = arith.mulf %val5, %scale : f32
        memref.store %res5, %out[%y, %idx5] : memref<1080x1920xf32>
        %c_off6 = arith.constant 6 : index
        %idx6 = arith.addi %x, %c_off6 : index
        %val6 = memref.load %in[%y, %idx6] : memref<1080x1920xf32>
        %res6 = arith.mulf %val6, %scale : f32
        memref.store %res6, %out[%y, %idx6] : memref<1080x1920xf32>
        %c_off7 = arith.constant 7 : index
        %idx7 = arith.addi %x, %c_off7 : index
        %val7 = memref.load %in[%y, %idx7] : memref<1080x1920xf32>
        %res7 = arith.mulf %val7, %scale : f32
        memref.store %res7, %out[%y, %idx7] : memref<1080x1920xf32>
      }
    }
    return
  }
}
