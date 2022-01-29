// Joint Bilateral Filter Kernel in MLIR
// Edge-preserving smoothing filter combining spatial Gaussian proximity
// and photometric radiometric intensity difference weights.

module @bilateral_filter {
  func.func @bilateral_filter_2d(
      %input: memref<?x?xf32>,
      %output: memref<?x?xf32>,
      %radius: index,
      %inv_two_sigma_s2: f32,
      %inv_two_sigma_r2: f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %f0 = arith.constant 0.000000e+00 : f32
    %f1 = arith.constant 1.000000e+00 : f32

    %height = memref.dim %input, %c0 : memref<?x?xf32>
    %width = memref.dim %input, %c1 : memref<?x?xf32>

    %h_bound = arith.subi %height, %radius : index
    %w_bound = arith.subi %width, %radius : index

    scf.for %y = %radius to %h_bound step %c1 {
      %y_i32 = arith.index_cast %y : index to i32
      %y_f = arith.sitofp %y_i32 : i32 to f32

      scf.for %x = %radius to %w_bound step %c1 {
        %x_i32 = arith.index_cast %x : index to i32
        %x_f = arith.sitofp %x_i32 : i32 to f32
        %center_val = memref.load %input[%y, %x] : memref<?x?xf32>

        %y_start = arith.subi %y, %radius : index
        %y_end = arith.addi %y, %radius : index
        %y_end1 = arith.addi %y_end, %c1 : index

        %x_start = arith.subi %x, %radius : index
        %x_end = arith.addi %x, %radius : index
        %x_end1 = arith.addi %x_end, %c1 : index

        // Accumulators for weighted sum and normalization weight
        %w_init = arith.constant 0.000000e+00 : f32
        %sum_init = arith.constant 0.000000e+00 : f32

        %sum_final, %w_final = scf.for %ky = %y_start to %y_end1 step %c1 iter_args(%s_acc = %sum_init, %w_acc = %w_init) -> (f32, f32) {
          %ky_i32 = arith.index_cast %ky : index to i32
          %ky_f = arith.sitofp %ky_i32 : i32 to f32
          %dy = arith.subf %ky_f, %y_f : f32
          %dy2 = arith.mulf %dy, %dy : f32

          %s_row, %w_row = scf.for %kx = %x_start to %x_end1 step %c1 iter_args(%s_in = %s_acc, %w_in = %w_acc) -> (f32, f32) {
            %kx_i32 = arith.index_cast %kx : index to i32
            %kx_f = arith.sitofp %kx_i32 : i32 to f32
            %dx = arith.subf %kx_f, %x_f : f32
            %dx2 = arith.mulf %dx, %dx : f32

            %spatial_dist2 = arith.addf %dx2, %dy2 : f32
            %spatial_exp_arg = arith.mulf %spatial_dist2, %inv_two_sigma_s2 : f32
            %spatial_neg_arg = arith.negf %spatial_exp_arg : f32
            %w_spatial = math.exp %spatial_neg_arg : f32

            %sample_val = memref.load %input[%ky, %kx] : memref<?x?xf32>
            %diff = arith.subf %sample_val, %center_val : f32
            %range_dist2 = arith.mulf %diff, %diff : f32
            %range_exp_arg = arith.mulf %range_dist2, %inv_two_sigma_r2 : f32
            %range_neg_arg = arith.negf %range_exp_arg : f32
            %w_range = math.exp %range_neg_arg : f32

            %w_total = arith.mulf %w_spatial, %w_range : f32
            %sample_weighted = arith.mulf %sample_val, %w_total : f32

            %s_next = arith.addf %s_in, %sample_weighted : f32
            %w_next = arith.addf %w_in, %w_total : f32
            scf.yield %s_next, %w_next : f32, f32
          }
          scf.yield %s_row, %w_row : f32, f32
        }

        %out_raw = arith.divf %sum_final, %w_final : f32
        %out_clamped1 = arith.maximumf %out_raw, %f0 : f32
        %out_val = arith.minimumf %out_clamped1, %f1 : f32
        memref.store %out_val, %output[%y, %x] : memref<?x?xf32>
      }
    }
    return
  }
}
