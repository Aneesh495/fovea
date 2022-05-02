// End-to-End Image Signal Processor Pipeline
// CFA Layout: RGGB, Demosaic Algorithm: bilinear, Resolution: 1440x2560

module @isp_pipeline_rggb_bilinear_1440x2560 {
  func.func @execute_pipeline(
      %raw_bayer: memref<1440x2560xf32>,
      %final_srgb: memref<1440x2560x3xf32>,
      %bl_offset: f32,
      %bl_scale: f32,
      %cx: f32, %cy: f32, %inv_r: f32,
      %k0: f32, %k1: f32, %k2: f32, %k3: f32,
      %gain_r: f32, %gain_g: f32, %gain_b: f32,
      %m00: f32, %m01: f32, %m02: f32,
      %m10: f32, %m11: f32, %m12: f32,
      %m20: f32, %m21: f32, %m22: f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %f0 = arith.constant 0.000000e+00 : f32
    %f1 = arith.constant 1.000000e+00 : f32
    %c_inv8 = arith.constant 0.125000e+00 : f32
    %c4_f = arith.constant 4.000000e+00 : f32
    %c2_f = arith.constant 2.000000e+00 : f32
    %c6_f = arith.constant 6.000000e+00 : f32
    %c1_5_f = arith.constant 1.500000e+00 : f32
    %c5_f = arith.constant 5.000000e+00 : f32
    %c0_5_f = arith.constant 0.500000e+00 : f32

    %aces_a = arith.constant 2.510000e+00 : f32
    %aces_b = arith.constant 0.030000e+00 : f32
    %aces_c = arith.constant 2.430000e+00 : f32
    %aces_d = arith.constant 0.590000e+00 : f32
    %aces_e = arith.constant 0.140000e+00 : f32

    %h_bound = arith.constant 1438 : index
    %w_bound = arith.constant 2558 : index

    scf.for %y = %c2 to %h_bound step %c1 {
      %y_i32 = arith.index_cast %y : index to i32
      %y_f = arith.sitofp %y_i32 : i32 to f32
      %dy = arith.subf %y_f, %cy : f32
      %dy2 = arith.mulf %dy, %dy : f32

      %y_prev1 = arith.subi %y, %c1 : index
      %y_next1 = arith.addi %y, %c1 : index
      %y_prev2 = arith.subi %y, %c2 : index
      %y_next2 = arith.addi %y, %c2 : index

      %y_rem = arith.remui %y, %c2 : index
      %is_even_row = arith.cmpi eq, %y_rem, %c0 : index

      scf.for %x = %c2 to %w_bound step %c1 {
        %x_i32 = arith.index_cast %x : index to i32
        %x_f = arith.sitofp %x_i32 : i32 to f32
        %dx = arith.subf %x_f, %cx : f32
        %dx2 = arith.mulf %dx, %dx : f32

        // Radial lens shading polynomial
        %r2_u = arith.addf %dx2, %dy2 : f32
        %r_u = math.sqrt %r2_u : f32
        %r_norm = arith.mulf %r_u, %inv_r : f32
        %rn2 = arith.mulf %r_norm, %r_norm : f32
        %rn4 = arith.mulf %rn2, %rn2 : f32
        %rn6 = arith.mulf %rn4, %rn2 : f32
        %term1 = arith.mulf %k1, %rn2 : f32
        %term2 = arith.mulf %k2, %rn4 : f32
        %term3 = arith.mulf %k3, %rn6 : f32
        %g01 = arith.addf %k0, %term1 : f32
        %g012 = arith.addf %g01, %term2 : f32
        %lens_gain = arith.addf %g012, %term3 : f32

        %x_prev1 = arith.subi %x, %c1 : index
        %x_next1 = arith.addi %x, %c1 : index
        %x_prev2 = arith.subi %x, %c2 : index
        %x_next2 = arith.addi %x, %c2 : index
        %x_rem = arith.remui %x, %c2 : index
        %is_even_col = arith.cmpi eq, %x_rem, %c0 : index

        // Linearized pixel loads
        %raw_c = memref.load %raw_bayer[%y, %x] : memref<1440x2560xf32>
        %lin_c_sub = arith.subf %raw_c, %bl_offset : f32
        %lin_c_sc = arith.mulf %lin_c_sub, %bl_scale : f32
        %center = arith.mulf %lin_c_sc, %lens_gain : f32

        %raw_t1 = memref.load %raw_bayer[%y_prev1, %x] : memref<1440x2560xf32>
        %top1 = arith.mulf %raw_t1, %bl_scale : f32
        %raw_b1 = memref.load %raw_bayer[%y_next1, %x] : memref<1440x2560xf32>
        %bot1 = arith.mulf %raw_b1, %bl_scale : f32
        %raw_l1 = memref.load %raw_bayer[%y, %x_prev1] : memref<1440x2560xf32>
        %left1 = arith.mulf %raw_l1, %bl_scale : f32
        %raw_r1 = memref.load %raw_bayer[%y, %x_next1] : memref<1440x2560xf32>
        %right1 = arith.mulf %raw_r1, %bl_scale : f32

        %raw_t2 = memref.load %raw_bayer[%y_prev2, %x] : memref<1440x2560xf32>
        %top2 = arith.mulf %raw_t2, %bl_scale : f32
        %raw_b2 = memref.load %raw_bayer[%y_next2, %x] : memref<1440x2560xf32>
        %bot2 = arith.mulf %raw_b2, %bl_scale : f32
        %raw_l2 = memref.load %raw_bayer[%y, %x_prev2] : memref<1440x2560xf32>
        %left2 = arith.mulf %raw_l2, %bl_scale : f32
        %raw_r2 = memref.load %raw_bayer[%y, %x_next2] : memref<1440x2560xf32>
        %right2 = arith.mulf %raw_r2, %bl_scale : f32

        %raw_tl = memref.load %raw_bayer[%y_prev1, %x_prev1] : memref<1440x2560xf32>
        %tl = arith.mulf %raw_tl, %bl_scale : f32
        %raw_tr = memref.load %raw_bayer[%y_prev1, %x_next1] : memref<1440x2560xf32>
        %tr = arith.mulf %raw_tr, %bl_scale : f32
        %raw_bl = memref.load %raw_bayer[%y_next1, %x_prev1] : memref<1440x2560xf32>
        %bl = arith.mulf %raw_bl, %bl_scale : f32
        %raw_br = memref.load %raw_bayer[%y_next1, %x_next1] : memref<1440x2560xf32>
        %br = arith.mulf %raw_br, %bl_scale : f32

        %sum_d1 = arith.addf %top1, %bot1 : f32
        %sum_d1_lr = arith.addf %left1, %right1 : f32
        %cross1 = arith.addf %sum_d1, %sum_d1_lr : f32

        %sum_d2 = arith.addf %top2, %bot2 : f32
        %sum_d2_lr = arith.addf %left2, %right2 : f32
        %cross2 = arith.addf %sum_d2, %sum_d2_lr : f32

        %sum_diag = arith.addf %tl, %tr : f32
        %sum_diag_b = arith.addf %bl, %br : f32
        %corners = arith.addf %sum_diag, %sum_diag_b : f32

        // Optimal CFA interpolation
        %dm_r, %dm_g, %dm_b = scf.if %is_even_row -> (f32, f32, f32) {
          %rr, %gg, %bb = scf.if %is_even_col -> (f32, f32, f32) {
            %g_t1 = arith.mulf %center, %c4_f : f32
            %g_t2 = arith.mulf %cross1, %c2_f : f32
            %g_a1 = arith.addf %g_t1, %g_t2 : f32
            %g_a2 = arith.subf %g_a1, %cross2 : f32
            %gv = arith.mulf %g_a2, %c_inv8 : f32

            %b_t1 = arith.mulf %center, %c6_f : f32
            %b_t2 = arith.mulf %corners, %c2_f : f32
            %b_t3 = arith.mulf %cross2, %c1_5_f : f32
            %b_a1 = arith.addf %b_t1, %b_t2 : f32
            %b_a2 = arith.subf %b_a1, %b_t3 : f32
            %bv = arith.mulf %b_a2, %c_inv8 : f32
            scf.yield %center, %gv, %bv : f32, f32, f32
          } else {
            %r_t1 = arith.mulf %center, %c5_f : f32
            %r_t2 = arith.mulf %sum_d1_lr, %c4_f : f32
            %r_a1 = arith.addf %r_t1, %r_t2 : f32
            %r_a2 = arith.subf %r_a1, %sum_d2_lr : f32
            %r_a3 = arith.subf %r_a2, %corners : f32
            %r_t4 = arith.mulf %sum_d2, %c0_5_f : f32
            %r_a4 = arith.addf %r_a3, %r_t4 : f32
            %rv = arith.mulf %r_a4, %c_inv8 : f32

            %b_t1 = arith.mulf %center, %c5_f : f32
            %b_t2 = arith.mulf %sum_d1, %c4_f : f32
            %b_a1 = arith.addf %b_t1, %b_t2 : f32
            %b_a2 = arith.subf %b_a1, %sum_d2 : f32
            %b_a3 = arith.subf %b_a2, %corners : f32
            %b_t4 = arith.mulf %sum_d2_lr, %c0_5_f : f32
            %b_a4 = arith.addf %b_a3, %b_t4 : f32
            %bv = arith.mulf %b_a4, %c_inv8 : f32
            scf.yield %rv, %center, %bv : f32, f32, f32
          }
          scf.yield %rr, %gg, %bb : f32, f32, f32
        } else {
          %rr, %gg, %bb = scf.if %is_even_col -> (f32, f32, f32) {
            %r_t1 = arith.mulf %center, %c5_f : f32
            %r_t2 = arith.mulf %sum_d1, %c4_f : f32
            %r_a1 = arith.addf %r_t1, %r_t2 : f32
            %r_a2 = arith.subf %r_a1, %sum_d2 : f32
            %r_a3 = arith.subf %r_a2, %corners : f32
            %r_t4 = arith.mulf %sum_d2_lr, %c0_5_f : f32
            %r_a4 = arith.addf %r_a3, %r_t4 : f32
            %rv = arith.mulf %r_a4, %c_inv8 : f32

            %b_t1 = arith.mulf %center, %c5_f : f32
            %b_t2 = arith.mulf %sum_d1_lr, %c4_f : f32
            %b_a1 = arith.addf %b_t1, %b_t2 : f32
            %b_a2 = arith.subf %b_a1, %sum_d2_lr : f32
            %b_a3 = arith.subf %b_a2, %corners : f32
            %b_t4 = arith.mulf %sum_d2, %c0_5_f : f32
            %b_a4 = arith.addf %b_a3, %b_t4 : f32
            %bv = arith.mulf %b_a4, %c_inv8 : f32
            scf.yield %rv, %center, %bv : f32, f32, f32
          } else {
            %g_t1 = arith.mulf %center, %c4_f : f32
            %g_t2 = arith.mulf %cross1, %c2_f : f32
            %g_a1 = arith.addf %g_t1, %g_t2 : f32
            %g_a2 = arith.subf %g_a1, %cross2 : f32
            %gv = arith.mulf %g_a2, %c_inv8 : f32

            %r_t1 = arith.mulf %center, %c6_f : f32
            %r_t2 = arith.mulf %corners, %c2_f : f32
            %r_t3 = arith.mulf %cross2, %c1_5_f : f32
            %r_a1 = arith.addf %r_t1, %r_t2 : f32
            %r_a2 = arith.subf %r_a1, %r_t3 : f32
            %rv = arith.mulf %r_a2, %c_inv8 : f32
            scf.yield %rv, %gv, %center : f32, f32, f32
          }
          scf.yield %rr, %gg, %bb : f32, f32, f32
        }

        // White balance adjustment
        %wb_r = arith.mulf %dm_r, %gain_r : f32
        %wb_g = arith.mulf %dm_g, %gain_g : f32
        %wb_b = arith.mulf %dm_b, %gain_b : f32

        // Color correction matrix transform
        %cr0 = arith.mulf %m00, %wb_r : f32
        %cr1 = arith.mulf %m01, %wb_g : f32
        %cr2 = arith.mulf %m02, %wb_b : f32
        %cr_s = arith.addf %cr0, %cr1 : f32
        %ccm_r = arith.addf %cr_s, %cr2 : f32

        %cg0 = arith.mulf %m10, %wb_r : f32
        %cg1 = arith.mulf %m11, %wb_g : f32
        %cg2 = arith.mulf %m12, %wb_b : f32
        %cg_s = arith.addf %cg0, %cg1 : f32
        %ccm_g = arith.addf %cg_s, %cg2 : f32

        %cb0 = arith.mulf %m20, %wb_r : f32
        %cb1 = arith.mulf %m21, %wb_g : f32
        %cb2 = arith.mulf %m22, %wb_b : f32
        %cb_s = arith.addf %cb0, %cb1 : f32
        %ccm_b = arith.addf %cb_s, %cb2 : f32

        // ACES Filmic Tone Mapping Curve
        %xr = arith.maximumf %ccm_r, %f0 : f32
        %axr = arith.mulf %aces_a, %xr : f32
        %axr_b = arith.addf %axr, %aces_b : f32
        %num_r = arith.mulf %xr, %axr_b : f32
        %cxr = arith.mulf %aces_c, %xr : f32
        %cxr_d = arith.addf %cxr, %aces_d : f32
        %dr_term = arith.mulf %xr, %cxr_d : f32
        %denom_r = arith.addf %dr_term, %aces_e : f32
        %aces_out_r = arith.divf %num_r, %denom_r : f32

        %xg = arith.maximumf %ccm_g, %f0 : f32
        %axg = arith.mulf %aces_a, %xg : f32
        %axg_b = arith.addf %axg, %aces_b : f32
        %num_g = arith.mulf %xg, %axg_b : f32
        %cxg = arith.mulf %aces_c, %xg : f32
        %cxg_d = arith.addf %cxg, %aces_d : f32
        %dg_term = arith.mulf %xg, %cxg_d : f32
        %denom_g = arith.addf %dg_term, %aces_e : f32
        %aces_out_g = arith.divf %num_g, %denom_g : f32

        %xb = arith.maximumf %ccm_b, %f0 : f32
        %axb = arith.mulf %aces_a, %xb : f32
        %axb_b = arith.addf %axb, %aces_b : f32
        %num_b = arith.mulf %xb, %axb_b : f32
        %cxb = arith.mulf %aces_c, %xb : f32
        %cxb_d = arith.addf %cxb, %aces_d : f32
        %db_term = arith.mulf %xb, %cxb_d : f32
        %denom_b = arith.addf %db_term, %aces_e : f32
        %aces_out_b = arith.divf %num_b, %denom_b : f32

        %fr_c = arith.maximumf %aces_out_r, %f0 : f32
        %final_r = arith.minimumf %fr_c, %f1 : f32
        %fg_c = arith.maximumf %aces_out_g, %f0 : f32
        %final_g = arith.minimumf %fg_c, %f1 : f32
        %fb_c = arith.maximumf %aces_out_b, %f0 : f32
        %final_b = arith.minimumf %fb_c, %f1 : f32

        %ch0 = arith.constant 0 : index
        %ch1 = arith.constant 1 : index
        %ch2 = arith.constant 2 : index
        memref.store %final_r, %final_srgb[%y, %x, %ch0] : memref<1440x2560x3xf32>
        memref.store %final_g, %final_srgb[%y, %x, %ch1] : memref<1440x2560x3xf32>
        memref.store %final_b, %final_srgb[%y, %x, %ch2] : memref<1440x2560x3xf32>
      }
    }
    return
  }
}
