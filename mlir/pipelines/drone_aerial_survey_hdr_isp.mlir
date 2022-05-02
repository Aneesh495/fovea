// High-Altitude Autonomous Drone Photogrammetry HDR Pipeline
// Domain-Specific Computational Photography Pipeline in MLIR

module @drone_aerial_survey_hdr_isp {
  func.func @pipeline_stage_calibration(
      %sensor_raw: memref<1080x1920xf32>,
      %calibrated: memref<1080x1920xf32>,
      %pedestal: f32,
      %scale: f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %h = arith.constant 1080 : index
    %w = arith.constant 1920 : index
    %f0 = arith.constant 0.000000e+00 : f32
    %f1 = arith.constant 1.000000e+00 : f32

    scf.for %y = %c0 to %h step %c1 {
      scf.for %x = %c0 to %w step %c1 {
        %v = memref.load %sensor_raw[%y, %x] : memref<1080x1920xf32>
        %sub = arith.subf %v, %pedestal : f32
        %lin = arith.mulf %sub, %scale : f32
        %c1_val = arith.maximumf %lin, %f0 : f32
        %out = arith.minimumf %c1_val, %f1 : f32
        memref.store %out, %calibrated[%y, %x] : memref<1080x1920xf32>
      }
    }
    return
  }

  func.func @pipeline_stage_demosaic_mhc(
      %bayer: memref<1080x1920xf32>,
      %rgb: memref<1080x1920x3xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %h_bound = arith.constant 1078 : index
    %w_bound = arith.constant 1918 : index
    %c_inv8 = arith.constant 0.125000e+00 : f32
    %c4_f = arith.constant 4.000000e+00 : f32
    %c2_f = arith.constant 2.000000e+00 : f32
    %c6_f = arith.constant 6.000000e+00 : f32
    %c1_5_f = arith.constant 1.500000e+00 : f32
    %c5_f = arith.constant 5.000000e+00 : f32
    %c0_5_f = arith.constant 0.500000e+00 : f32
    %f0 = arith.constant 0.000000e+00 : f32
    %f1 = arith.constant 1.000000e+00 : f32

    scf.for %y = %c2 to %h_bound step %c1 {
      %y_prev1 = arith.subi %y, %c1 : index
      %y_next1 = arith.addi %y, %c1 : index
      %y_prev2 = arith.subi %y, %c2 : index
      %y_next2 = arith.addi %y, %c2 : index
      %y_rem = arith.remui %y, %c2 : index
      %is_even_row = arith.cmpi eq, %y_rem, %c0 : index

      scf.for %x = %c2 to %w_bound step %c1 {
        %x_prev1 = arith.subi %x, %c1 : index
        %x_next1 = arith.addi %x, %c1 : index
        %x_prev2 = arith.subi %x, %c2 : index
        %x_next2 = arith.addi %x, %c2 : index
        %x_rem = arith.remui %x, %c2 : index
        %is_even_col = arith.cmpi eq, %x_rem, %c0 : index

        %center = memref.load %bayer[%y, %x] : memref<1080x1920xf32>
        %top1 = memref.load %bayer[%y_prev1, %x] : memref<1080x1920xf32>
        %bot1 = memref.load %bayer[%y_next1, %x] : memref<1080x1920xf32>
        %left1 = memref.load %bayer[%y, %x_prev1] : memref<1080x1920xf32>
        %right1 = memref.load %bayer[%y, %x_next1] : memref<1080x1920xf32>

        %top2 = memref.load %bayer[%y_prev2, %x] : memref<1080x1920xf32>
        %bot2 = memref.load %bayer[%y_next2, %x] : memref<1080x1920xf32>
        %left2 = memref.load %bayer[%y, %x_prev2] : memref<1080x1920xf32>
        %right2 = memref.load %bayer[%y, %x_next2] : memref<1080x1920xf32>

        %tl = memref.load %bayer[%y_prev1, %x_prev1] : memref<1080x1920xf32>
        %tr = memref.load %bayer[%y_prev1, %x_next1] : memref<1080x1920xf32>
        %bl = memref.load %bayer[%y_next1, %x_prev1] : memref<1080x1920xf32>
        %br = memref.load %bayer[%y_next1, %x_next1] : memref<1080x1920xf32>

        %sum_d1 = arith.addf %top1, %bot1 : f32
        %sum_d1_lr = arith.addf %left1, %right1 : f32
        %cross1 = arith.addf %sum_d1, %sum_d1_lr : f32

        %sum_d2 = arith.addf %top2, %bot2 : f32
        %sum_d2_lr = arith.addf %left2, %right2 : f32
        %cross2 = arith.addf %sum_d2, %sum_d2_lr : f32

        %sum_diag = arith.addf %tl, %tr : f32
        %sum_diag_b = arith.addf %bl, %br : f32
        %corners = arith.addf %sum_diag, %sum_diag_b : f32

        %r, %g, %b = scf.if %is_even_row -> (f32, f32, f32) {
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

        %r_c1 = arith.maximumf %r, %f0 : f32
        %r_out = arith.minimumf %r_c1, %f1 : f32
        %g_c1 = arith.maximumf %g, %f0 : f32
        %g_out = arith.minimumf %g_c1, %f1 : f32
        %b_c1 = arith.maximumf %b, %f0 : f32
        %b_out = arith.minimumf %b_c1, %f1 : f32

        %ch0 = arith.constant 0 : index
        %ch1 = arith.constant 1 : index
        %ch2 = arith.constant 2 : index
        memref.store %r_out, %rgb[%y, %x, %ch0] : memref<1080x1920x3xf32>
        memref.store %g_out, %rgb[%y, %x, %ch1] : memref<1080x1920x3xf32>
        memref.store %b_out, %rgb[%y, %x, %ch2] : memref<1080x1920x3xf32>
      }
    }
    return
  }

  func.func @pipeline_stage_color_correction(
      %rgb_in: memref<1080x1920x3xf32>,
      %rgb_out: memref<1080x1920x3xf32>,
      %m00: f32, %m01: f32, %m02: f32,
      %m10: f32, %m11: f32, %m12: f32,
      %m20: f32, %m21: f32, %m22: f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %h = arith.constant 1080 : index
    %w = arith.constant 1920 : index
    %ch0 = arith.constant 0 : index
    %ch1 = arith.constant 1 : index
    %ch2 = arith.constant 2 : index
    %f0 = arith.constant 0.000000e+00 : f32
    %f1 = arith.constant 1.000000e+00 : f32

    scf.for %y = %c0 to %h step %c1 {
      scf.for %x = %c0 to %w step %c1 {
        %r = memref.load %rgb_in[%y, %x, %ch0] : memref<1080x1920x3xf32>
        %g = memref.load %rgb_in[%y, %x, %ch1] : memref<1080x1920x3xf32>
        %b = memref.load %rgb_in[%y, %x, %ch2] : memref<1080x1920x3xf32>

        %r0 = arith.mulf %m00, %r : f32
        %r1 = arith.mulf %m01, %g : f32
        %r2 = arith.mulf %m02, %b : f32
        %rs = arith.addf %r0, %r1 : f32
        %ro = arith.addf %rs, %r2 : f32

        %g0 = arith.mulf %m10, %r : f32
        %g1 = arith.mulf %m11, %g : f32
        %g2 = arith.mulf %m12, %b : f32
        %gs = arith.addf %g0, %g1 : f32
        %go = arith.addf %gs, %g2 : f32

        %b0 = arith.mulf %m20, %r : f32
        %b1 = arith.mulf %m21, %g : f32
        %b2 = arith.mulf %m22, %b : f32
        %bs = arith.addf %b0, %b1 : f32
        %bo = arith.addf %bs, %b2 : f32

        %rc = arith.maximumf %ro, %f0 : f32
        %rf = arith.minimumf %rc, %f1 : f32
        %gc = arith.maximumf %go, %f0 : f32
        %gf = arith.minimumf %gc, %f1 : f32
        %bc = arith.maximumf %bo, %f0 : f32
        %bf = arith.minimumf %bc, %f1 : f32

        memref.store %rf, %rgb_out[%y, %x, %ch0] : memref<1080x1920x3xf32>
        memref.store %gf, %rgb_out[%y, %x, %ch1] : memref<1080x1920x3xf32>
        memref.store %bf, %rgb_out[%y, %x, %ch2] : memref<1080x1920x3xf32>
      }
    }
    return
  }
}
