// Unrolled SCF Loop Nest: Filmic, Unroll 4x on 1080x1920
module @scf_unroll4_aces_1080x1920 {
  func.func @execute_unroll(
      %in: memref<1080x1920xf32>,
      %out: memref<1080x1920xf32>,
      %scale: f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %h_dim = arith.constant 1080 : index
    %w_dim = arith.constant 1920 : index
    %step = arith.constant 4 : index

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
      }
    }
    return
  }
}
