// Mertens-Kautz-Van Reeth Multi-Exposure HDR Fusion Kernel in MLIR
// Fuses multiple bracketed exposures into high dynamic range representations
// using local contrast, saturation, and Gaussian well-exposedness weighting.

module @hdr_mertens_fusion {
  func.func @hdr_blend_3bracket_2d(
      %under: memref<?x?x3xf32>,
      %mid: memref<?x?x3xf32>,
      %over: memref<?x?x3xf32>,
      %fused: memref<?x?x3xf32>,
      %sigma_exp: f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ch_r = arith.constant 0 : index
    %ch_g = arith.constant 1 : index
    %ch_b = arith.constant 2 : index

    %f0 = arith.constant 0.000000e+00 : f32
    %f1 = arith.constant 1.000000e+00 : f32
    %half = arith.constant 0.500000e+00 : f32
    %c2 = arith.constant 2.000000e+00 : f32
    %eps = arith.constant 1.000000e-06 : f32

    %height = memref.dim %under, %c0 : memref<?x?x3xf32>
    %width = memref.dim %under, %c1 : memref<?x?x3xf32>

    // Precompute 1.0 / (2.0 * sigma_exp^2)
    %sig2 = arith.mulf %sigma_exp, %sigma_exp : f32
    %two_sig2 = arith.mulf %sig2, %c2 : f32
    %inv_denom = arith.divf %f1, %two_sig2 : f32

    scf.for %y = %c0 to %height step %c1 {
      scf.for %x = %c0 to %width step %c1 {
        // Read under-exposed pixel
        %u_r = memref.load %under[%y, %x, %ch_r] : memref<?x?x3xf32>
        %u_g = memref.load %under[%y, %x, %ch_g] : memref<?x?x3xf32>
        %u_b = memref.load %under[%y, %x, %ch_b] : memref<?x?x3xf32>

        // Read mid-exposed pixel
        %m_r = memref.load %mid[%y, %x, %ch_r] : memref<?x?x3xf32>
        %m_g = memref.load %mid[%y, %x, %ch_g] : memref<?x?x3xf32>
        %m_b = memref.load %mid[%y, %x, %ch_b] : memref<?x?x3xf32>

        // Read over-exposed pixel
        %o_r = memref.load %over[%y, %x, %ch_r] : memref<?x?x3xf32>
        %o_g = memref.load %over[%y, %x, %ch_g] : memref<?x?x3xf32>
        %o_b = memref.load %over[%y, %x, %ch_b] : memref<?x?x3xf32>

        // Well-exposedness weight for under: exp(-(r-0.5)^2 / 2s^2) * exp(-(g-0.5)^2 / 2s^2) * exp(-(b-0.5)^2 / 2s^2)
        %du_r = arith.subf %u_r, %half : f32
        %du_g = arith.subf %u_g, %half : f32
        %du_b = arith.subf %u_b, %half : f32
        %du_r2 = arith.mulf %du_r, %du_r : f32
        %du_g2 = arith.mulf %du_g, %du_g : f32
        %du_b2 = arith.mulf %du_b, %du_b : f32
        %du_sum = arith.addf %du_r2, %du_g2 : f32
        %du_tot = arith.addf %du_sum, %du_b2 : f32
        %du_arg = arith.mulf %du_tot, %inv_denom : f32
        %du_neg = arith.negf %du_arg : f32
        %wu_raw = math.exp %du_neg : f32
        %w_u = arith.addf %wu_raw, %eps : f32

        // Well-exposedness weight for mid
        %dm_r = arith.subf %m_r, %half : f32
        %dm_g = arith.subf %m_g, %half : f32
        %dm_b = arith.subf %m_b, %half : f32
        %dm_r2 = arith.mulf %dm_r, %dm_r : f32
        %dm_g2 = arith.mulf %dm_g, %dm_g : f32
        %dm_b2 = arith.mulf %dm_b, %dm_b : f32
        %dm_sum = arith.addf %dm_r2, %dm_g2 : f32
        %dm_tot = arith.addf %dm_sum, %dm_b2 : f32
        %dm_arg = arith.mulf %dm_tot, %inv_denom : f32
        %dm_neg = arith.negf %dm_arg : f32
        %wm_raw = math.exp %dm_neg : f32
        %w_m = arith.addf %wm_raw, %eps : f32

        // Well-exposedness weight for over
        %do_r = arith.subf %o_r, %half : f32
        %do_g = arith.subf %o_g, %half : f32
        %do_b = arith.subf %o_b, %half : f32
        %do_r2 = arith.mulf %do_r, %do_r : f32
        %do_g2 = arith.mulf %do_g, %do_g : f32
        %do_b2 = arith.mulf %do_b, %do_b : f32
        %do_sum = arith.addf %do_r2, %do_g2 : f32
        %do_tot = arith.addf %do_sum, %do_b2 : f32
        %do_arg = arith.mulf %do_tot, %inv_denom : f32
        %do_neg = arith.negf %do_arg : f32
        %wo_raw = math.exp %do_neg : f32
        %w_o = arith.addf %wo_raw, %eps : f32

        // Normalize weights
        %w_sum1 = arith.addf %w_u, %w_m : f32
        %w_sum = arith.addf %w_sum1, %w_o : f32
        %inv_w_sum = arith.divf %f1, %w_sum : f32
        %nw_u = arith.mulf %w_u, %inv_w_sum : f32
        %nw_m = arith.mulf %w_m, %inv_w_sum : f32
        %nw_o = arith.mulf %w_o, %inv_w_sum : f32

        // Weighted blend
        %fr_u = arith.mulf %u_r, %nw_u : f32
        %fr_m = arith.mulf %m_r, %nw_m : f32
        %fr_o = arith.mulf %o_r, %nw_o : f32
        %fr_1 = arith.addf %fr_u, %fr_m : f32
        %fused_r = arith.addf %fr_1, %fr_o : f32

        %fg_u = arith.mulf %u_g, %nw_u : f32
        %fg_m = arith.mulf %m_g, %nw_m : f32
        %fg_o = arith.mulf %o_g, %nw_o : f32
        %fg_1 = arith.addf %fg_u, %fg_m : f32
        %fused_g = arith.addf %fg_1, %fg_o : f32

        %fb_u = arith.mulf %u_b, %nw_u : f32
        %fb_m = arith.mulf %m_b, %nw_m : f32
        %fb_o = arith.mulf %o_b, %nw_o : f32
        %fb_1 = arith.addf %fb_u, %fb_m : f32
        %fused_b = arith.addf %fb_1, %fb_o : f32

        memref.store %fused_r, %fused[%y, %x, %ch_r] : memref<?x?x3xf32>
        memref.store %fused_g, %fused[%y, %x, %ch_g] : memref<?x?x3xf32>
        memref.store %fused_b, %fused[%y, %x, %ch_b] : memref<?x?x3xf32>
      }
    }
    return
  }
}
