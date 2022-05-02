// Burst HDR Multi-Exposure Fusion Pipeline in MLIR
module @burst_hdr_pipeline {
  func.func @burst_hdr_3bracket(
      %under: memref<1080x1920x3xf32>,
      %mid: memref<1080x1920x3xf32>,
      %over: memref<1080x1920x3xf32>,
      %fused: memref<1080x1920x3xf32>,
      %sigma: f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %h = arith.constant 1080 : index
    %w = arith.constant 1920 : index
    %f1 = arith.constant 1.000000e+00 : f32
    %half = arith.constant 0.500000e+00 : f32
    %c2 = arith.constant 2.000000e+00 : f32
    %sig2 = arith.mulf %sigma, %sigma : f32
    %two_sig2 = arith.mulf %sig2, %c2 : f32
    %inv_d = arith.divf %f1, %two_sig2 : f32

    scf.for %y = %c0 to %h step %c1 {
      scf.for %x = %c0 to %w step %c1 {
        scf.for %c = %c0 to %c1 step %c1 {
          %u = memref.load %under[%y, %x, %c] : memref<1080x1920x3xf32>
          %m = memref.load %mid[%y, %x, %c] : memref<1080x1920x3xf32>
          %o = memref.load %over[%y, %x, %c] : memref<1080x1920x3xf32>

          %du = arith.subf %u, %half : f32
          %du2 = arith.mulf %du, %du : f32
          %arg_u = arith.mulf %du2, %inv_d : f32
          %neg_u = arith.negf %arg_u : f32
          %wu = math.exp %neg_u : f32

          %dm = arith.subf %m, %half : f32
          %dm2 = arith.mulf %dm, %dm : f32
          %arg_m = arith.mulf %dm2, %inv_d : f32
          %neg_m = arith.negf %arg_m : f32
          %wm = math.exp %neg_m : f32

          %do = arith.subf %o, %half : f32
          %do2 = arith.mulf %do, %do : f32
          %arg_o = arith.mulf %do2, %inv_d : f32
          %neg_o = arith.negf %arg_o : f32
          %wo = math.exp %neg_o : f32

          %w_sum = arith.addf %wu, %wm : f32
          %w_tot = arith.addf %w_sum, %wo : f32
          %inv_w = arith.divf %f1, %w_tot : f32

          %blend1 = arith.mulf %u, %wu : f32
          %blend2 = arith.mulf %m, %wm : f32
          %blend3 = arith.mulf %o, %wo : f32
          %b_sum = arith.addf %blend1, %blend2 : f32
          %b_tot = arith.addf %b_sum, %blend3 : f32
          %fused_val = arith.mulf %b_tot, %inv_w : f32

          memref.store %fused_val, %fused[%y, %x, %c] : memref<1080x1920x3xf32>
        }
      }
    }
    return
  }
}
