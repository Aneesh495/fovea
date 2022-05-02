// Multi-Exposure HDR Exposure Fusion: 2-Bracket Fusion on 1440x2560
module @hdr_fusion_2bracket_1440x2560 {
  func.func @fuse_2bracket(
      %bracket0: memref<1440x2560x3xf32>,
      %bracket1: memref<1440x2560x3xf32>,
      %fused: memref<1440x2560x3xf32>,
      %sigma: f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %h_dim = arith.constant 1440 : index
    %w_dim = arith.constant 2560 : index
    %f1 = arith.constant 1.000000e+00 : f32
    %half = arith.constant 0.500000e+00 : f32
    %c2 = arith.constant 2.000000e+00 : f32
    %sig2 = arith.mulf %sigma, %sigma : f32
    %two_sig2 = arith.mulf %sig2, %c2 : f32
    %inv_d = arith.divf %f1, %two_sig2 : f32

    scf.for %y = %c0 to %h_dim step %c1 {
      scf.for %x = %c0 to %w_dim step %c1 {
        scf.for %c = %c0 to %c1 step %c1 {
          %v0 = memref.load %bracket0[%y, %x, %c] : memref<1440x2560x3xf32>
          %v1 = memref.load %bracket1[%y, %x, %c] : memref<1440x2560x3xf32>

          %d0 = arith.subf %v0, %half : f32
          %d0_2 = arith.mulf %d0, %d0 : f32
          %arg0 = arith.mulf %d0_2, %inv_d : f32
          %neg0 = arith.negf %arg0 : f32
          %w0 = math.exp %neg0 : f32

          %d1 = arith.subf %v1, %half : f32
          %d1_2 = arith.mulf %d1, %d1 : f32
          %arg1 = arith.mulf %d1_2, %inv_d : f32
          %neg1 = arith.negf %arg1 : f32
          %w1 = math.exp %neg1 : f32

          %w_sum = arith.addf %w0, %w1 : f32
          %inv_w = arith.divf %f1, %w_sum : f32

          %b0 = arith.mulf %v0, %w0 : f32
          %b1 = arith.mulf %v1, %w1 : f32
          %b_tot = arith.addf %b0, %b1 : f32
          %res = arith.mulf %b_tot, %inv_w : f32

          memref.store %res, %fused[%y, %x, %c] : memref<1440x2560x3xf32>
        }
      }
    }
    return
  }
}
