// Directional Gradient-Adaptive Bayer CFA Demosaicing in MLIR
// Evaluates horizontal and vertical second-order spatial gradients and interpolates
// along the direction of minimum variation to suppress edge blurring and chromatic zipper artifacts.

module @demosaic_directional {
  func.func @demosaic_directional_rggb_2d(
      %bayer: memref<?x?xf32>,
      %rgb: memref<?x?x3xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %ch_r = arith.constant 0 : index
    %ch_g = arith.constant 1 : index
    %ch_b = arith.constant 2 : index

    %f0 = arith.constant 0.000000e+00 : f32
    %f1 = arith.constant 1.000000e+00 : f32
    %c_half = arith.constant 0.500000e+00 : f32
    %c_quarter = arith.constant 0.250000e+00 : f32
    %c2_f = arith.constant 2.000000e+00 : f32

    %height = memref.dim %bayer, %c0 : memref<?x?xf32>
    %width = memref.dim %bayer, %c1 : memref<?x?xf32>

    %h_sub2 = arith.subi %height, %c2 : index
    %w_sub2 = arith.subi %width, %c2 : index

    scf.for %y = %c2 to %h_sub2 step %c1 {
      %y_prev1 = arith.subi %y, %c1 : index
      %y_next1 = arith.addi %y, %c1 : index
      %y_prev2 = arith.subi %y, %c2 : index
      %y_next2 = arith.addi %y, %c2 : index

      %y_rem = arith.remui %y, %c2 : index
      %is_even_row = arith.cmpi eq, %y_rem, %c0 : index

      scf.for %x = %c2 to %w_sub2 step %c1 {
        %x_prev1 = arith.subi %x, %c1 : index
        %x_next1 = arith.addi %x, %c1 : index
        %x_prev2 = arith.subi %x, %c2 : index
        %x_next2 = arith.addi %x, %c2 : index

        %x_rem = arith.remui %x, %c2 : index
        %is_even_col = arith.cmpi eq, %x_rem, %c0 : index

        %center = memref.load %bayer[%y, %x] : memref<?x?xf32>

        %top1 = memref.load %bayer[%y_prev1, %x] : memref<?x?xf32>
        %bot1 = memref.load %bayer[%y_next1, %x] : memref<?x?xf32>
        %left1 = memref.load %bayer[%y, %x_prev1] : memref<?x?xf32>
        %right1 = memref.load %bayer[%y, %x_next1] : memref<?x?xf32>

        %top2 = memref.load %bayer[%y_prev2, %x] : memref<?x?xf32>
        %bot2 = memref.load %bayer[%y_next2, %x] : memref<?x?xf32>
        %left2 = memref.load %bayer[%y, %x_prev2] : memref<?x?xf32>
        %right2 = memref.load %bayer[%y, %x_next2] : memref<?x?xf32>

        // Calculate horizontal gradient: |L1 - R1| + |2*C - L2 - R2|
        %diff_lr1 = arith.subf %left1, %right1 : f32
        %abs_diff_lr1 = math.absf %diff_lr1 : f32
        %c_2x = arith.mulf %center, %c2_f : f32
        %sum_lr2 = arith.addf %left2, %right2 : f32
        %diff_lr2 = arith.subf %c_2x, %sum_lr2 : f32
        %abs_diff_lr2 = math.absf %diff_lr2 : f32
        %grad_h = arith.addf %abs_diff_lr1, %abs_diff_lr2 : f32

        // Calculate vertical gradient: |T1 - B1| + |2*C - T2 - B2|
        %diff_tb1 = arith.subf %top1, %bot1 : f32
        %abs_diff_tb1 = math.absf %diff_tb1 : f32
        %sum_tb2 = arith.addf %top2, %bot2 : f32
        %diff_tb2 = arith.subf %c_2x, %sum_tb2 : f32
        %abs_diff_tb2 = math.absf %diff_tb2 : f32
        %grad_v = arith.addf %abs_diff_tb1, %abs_diff_tb2 : f32

        // Determine green interpolation along direction of minimal variation
        %h_less = arith.cmpf olt, %grad_h, %grad_v : f32
        %v_less = arith.cmpf olt, %grad_v, %grad_h : f32

        %g_lr_avg = arith.addf %left1, %right1 : f32
        %g_lr_half = arith.mulf %g_lr_avg, %c_half : f32
        %g_lr_lap = arith.mulf %diff_lr2, %c_quarter : f32
        %g_h = arith.addf %g_lr_half, %g_lr_lap : f32

        %g_tb_avg = arith.addf %top1, %bot1 : f32
        %g_tb_half = arith.mulf %g_tb_avg, %c_half : f32
        %g_tb_lap = arith.mulf %diff_tb2, %c_quarter : f32
        %g_v = arith.addf %g_tb_half, %g_tb_lap : f32

        %g_omni = arith.addf %g_lr_avg, %g_tb_avg : f32
        %g_diag = arith.mulf %g_omni, %c_quarter : f32

        %g_interp1 = arith.select %h_less, %g_h, %g_diag : f32
        %g_interp = arith.select %v_less, %g_v, %g_interp1 : f32

        // Corner blue/red averages
        %tl = memref.load %bayer[%y_prev1, %x_prev1] : memref<?x?xf32>
        %tr = memref.load %bayer[%y_prev1, %x_next1] : memref<?x?xf32>
        %bl = memref.load %bayer[%y_next1, %x_prev1] : memref<?x?xf32>
        %br = memref.load %bayer[%y_next1, %x_next1] : memref<?x?xf32>
        %c_sum1 = arith.addf %tl, %tr : f32
        %c_sum2 = arith.addf %bl, %br : f32
        %c_tot = arith.addf %c_sum1, %c_sum2 : f32
        %c_avg = arith.mulf %c_tot, %c_quarter : f32

        %r, %g, %b = scf.if %is_even_row -> (f32, f32, f32) {
          %r_row, %g_row, %b_row = scf.if %is_even_col -> (f32, f32, f32) {
            // R at R
            scf.yield %center, %g_interp, %c_avg : f32, f32, f32
          } else {
            // Gr at Gr
            scf.yield %g_lr_half, %center, %g_tb_half : f32, f32, f32
          }
          scf.yield %r_row, %g_row, %b_row : f32, f32, f32
        } else {
          %r_row, %g_row, %b_row = scf.if %is_even_col -> (f32, f32, f32) {
            // Gb at Gb
            scf.yield %g_tb_half, %center, %g_lr_half : f32, f32, f32
          } else {
            // B at B
            scf.yield %c_avg, %g_interp, %center : f32, f32, f32
          }
          scf.yield %r_row, %g_row, %b_row : f32, f32, f32
        }

        %r_clamped1 = arith.maximumf %r, %f0 : f32
        %r_clamped = arith.minimumf %r_clamped1, %f1 : f32
        %g_clamped1 = arith.maximumf %g, %f0 : f32
        %g_clamped = arith.minimumf %g_clamped1, %f1 : f32
        %b_clamped1 = arith.maximumf %b, %f0 : f32
        %b_clamped = arith.minimumf %b_clamped1, %f1 : f32

        memref.store %r_clamped, %rgb[%y, %x, %ch_r] : memref<?x?x3xf32>
        memref.store %g_clamped, %rgb[%y, %x, %ch_g] : memref<?x?x3xf32>
        memref.store %b_clamped, %rgb[%y, %x, %ch_b] : memref<?x?x3xf32>
      }
    }
    return
  }
}
