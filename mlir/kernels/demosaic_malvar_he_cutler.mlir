// Malvar-He-Cutler 5x5 High-Quality Linear FIR Demosaicing in MLIR
// Reconstructs missing color channels using 5x5 optimal FIR filter stencils
// with cross-channel high-frequency Laplacian gradient guidance.

module @demosaic_malvar_he_cutler {
  func.func @demosaic_mhc_rggb_2d(
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
    %c_inv8 = arith.constant 0.125000e+00 : f32
    %c2_f = arith.constant 2.000000e+00 : f32
    %c4_f = arith.constant 4.000000e+00 : f32
    %c5_f = arith.constant 5.000000e+00 : f32
    %c6_f = arith.constant 6.000000e+00 : f32
    %c1_5_f = arith.constant 1.500000e+00 : f32
    %c0_5_f = arith.constant 0.500000e+00 : f32

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

        %tl = memref.load %bayer[%y_prev1, %x_prev1] : memref<?x?xf32>
        %tr = memref.load %bayer[%y_prev1, %x_next1] : memref<?x?xf32>
        %bl = memref.load %bayer[%y_next1, %x_prev1] : memref<?x?xf32>
        %br = memref.load %bayer[%y_next1, %x_next1] : memref<?x?xf32>

        // Shared sums
        %sum_d1 = arith.addf %top1, %bot1 : f32
        %sum_d1_lr = arith.addf %left1, %right1 : f32
        %cross1 = arith.addf %sum_d1, %sum_d1_lr : f32

        %sum_d2 = arith.addf %top2, %bot2 : f32
        %sum_d2_lr = arith.addf %left2, %right2 : f32
        %cross2 = arith.addf %sum_d2, %sum_d2_lr : f32

        %sum_diag = arith.addf %tl, %tr : f32
        %sum_diag_b = arith.addf %bl, %br : f32
        %corners = arith.addf %sum_diag, %sum_diag_b : f32

        // Branch by Bayer CFA location
        %r, %g, %b = scf.if %is_even_row -> (f32, f32, f32) {
          %r_row, %g_row, %b_row = scf.if %is_even_col -> (f32, f32, f32) {
            // R pixel
            %g_term1 = arith.mulf %center, %c4_f : f32
            %g_term2 = arith.mulf %cross1, %c2_f : f32
            %g_acc1 = arith.addf %g_term1, %g_term2 : f32
            %g_acc2 = arith.subf %g_acc1, %cross2 : f32
            %g_val = arith.mulf %g_acc2, %c_inv8 : f32

            %b_term1 = arith.mulf %center, %c6_f : f32
            %b_term2 = arith.mulf %corners, %c2_f : f32
            %b_term3 = arith.mulf %cross2, %c1_5_f : f32
            %b_acc1 = arith.addf %b_term1, %b_term2 : f32
            %b_acc2 = arith.subf %b_acc1, %b_term3 : f32
            %b_val = arith.mulf %b_acc2, %c_inv8 : f32

            scf.yield %center, %g_val, %b_val : f32, f32, f32
          } else {
            // Gr pixel
            %r_term1 = arith.mulf %center, %c5_f : f32
            %r_term2 = arith.mulf %sum_d1_lr, %c4_f : f32
            %r_acc1 = arith.addf %r_term1, %r_term2 : f32
            %r_acc2 = arith.subf %r_acc1, %sum_d2_lr : f32
            %r_acc3 = arith.subf %r_acc2, %corners : f32
            %r_term4 = arith.mulf %sum_d2, %c0_5_f : f32
            %r_acc4 = arith.addf %r_acc3, %r_term4 : f32
            %r_val = arith.mulf %r_acc4, %c_inv8 : f32

            %b_term1 = arith.mulf %center, %c5_f : f32
            %b_term2 = arith.mulf %sum_d1, %c4_f : f32
            %b_acc1 = arith.addf %b_term1, %b_term2 : f32
            %b_acc2 = arith.subf %b_acc1, %sum_d2 : f32
            %b_acc3 = arith.subf %b_acc2, %corners : f32
            %b_term4 = arith.mulf %sum_d2_lr, %c0_5_f : f32
            %b_acc4 = arith.addf %b_acc3, %b_term4 : f32
            %b_val = arith.mulf %b_acc4, %c_inv8 : f32

            scf.yield %r_val, %center, %b_val : f32, f32, f32
          }
          scf.yield %r_row, %g_row, %b_row : f32, f32, f32
        } else {
          %r_row, %g_row, %b_row = scf.if %is_even_col -> (f32, f32, f32) {
            // Gb pixel
            %r_term1 = arith.mulf %center, %c5_f : f32
            %r_term2 = arith.mulf %sum_d1, %c4_f : f32
            %r_acc1 = arith.addf %r_term1, %r_term2 : f32
            %r_acc2 = arith.subf %r_acc1, %sum_d2 : f32
            %r_acc3 = arith.subf %r_acc2, %corners : f32
            %r_term4 = arith.mulf %sum_d2_lr, %c0_5_f : f32
            %r_acc4 = arith.addf %r_acc3, %r_term4 : f32
            %r_val = arith.mulf %r_acc4, %c_inv8 : f32

            %b_term1 = arith.mulf %center, %c5_f : f32
            %b_term2 = arith.mulf %sum_d1_lr, %c4_f : f32
            %b_acc1 = arith.addf %b_term1, %b_term2 : f32
            %b_acc2 = arith.subf %b_acc1, %sum_d2_lr : f32
            %b_acc3 = arith.subf %b_acc2, %corners : f32
            %b_term4 = arith.mulf %sum_d2, %c0_5_f : f32
            %b_acc4 = arith.addf %b_acc3, %b_term4 : f32
            %b_val = arith.mulf %b_acc4, %c_inv8 : f32

            scf.yield %r_val, %center, %b_val : f32, f32, f32
          } else {
            // B pixel
            %g_term1 = arith.mulf %center, %c4_f : f32
            %g_term2 = arith.mulf %cross1, %c2_f : f32
            %g_acc1 = arith.addf %g_term1, %g_term2 : f32
            %g_acc2 = arith.subf %g_acc1, %cross2 : f32
            %g_val = arith.mulf %g_acc2, %c_inv8 : f32

            %r_term1 = arith.mulf %center, %c6_f : f32
            %r_term2 = arith.mulf %corners, %c2_f : f32
            %r_term3 = arith.mulf %cross2, %c1_5_f : f32
            %r_acc1 = arith.addf %r_term1, %r_term2 : f32
            %r_acc2 = arith.subf %r_acc1, %r_term3 : f32
            %r_val = arith.mulf %r_acc2, %c_inv8 : f32

            scf.yield %r_val, %g_val, %center : f32, f32, f32
          }
          scf.yield %r_row, %g_row, %b_row : f32, f32, f32
        }

        // Clamp outputs to [0.0, 1.0]
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
