// Bilinear Bayer CFA Demosaicing Kernel in MLIR
// Reconstructs full 3-channel RGB from single-channel Bayer mosaic for RGGB layout.

module @demosaic_bilinear {
  func.func @demosaic_rggb_2d(
      %bayer: memref<?x?xf32>,
      %rgb: memref<?x?x3xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %ch_r = arith.constant 0 : index
    %ch_g = arith.constant 1 : index
    %ch_b = arith.constant 2 : index

    %c_half = arith.constant 0.500000e+00 : f32
    %c_quarter = arith.constant 0.250000e+00 : f32

    %height = memref.dim %bayer, %c0 : memref<?x?xf32>
    %width = memref.dim %bayer, %c1 : memref<?x?xf32>

    %h_sub1 = arith.subi %height, %c1 : index
    %w_sub1 = arith.subi %width, %c1 : index

    scf.for %y = %c1 to %h_sub1 step %c1 {
      %y_prev = arith.subi %y, %c1 : index
      %y_next = arith.addi %y, %c1 : index
      %y_rem = arith.remui %y, %c2 : index
      %is_even_row = arith.cmpi eq, %y_rem, %c0 : index

      scf.for %x = %c1 to %w_sub1 step %c1 {
        %x_prev = arith.subi %x, %c1 : index
        %x_next = arith.addi %x, %c1 : index
        %x_rem = arith.remui %x, %c2 : index
        %is_even_col = arith.cmpi eq, %x_rem, %c0 : index

        %center = memref.load %bayer[%y, %x] : memref<?x?xf32>
        %top = memref.load %bayer[%y_prev, %x] : memref<?x?xf32>
        %bot = memref.load %bayer[%y_next, %x] : memref<?x?xf32>
        %left = memref.load %bayer[%y, %x_prev] : memref<?x?xf32>
        %right = memref.load %bayer[%y, %x_next] : memref<?x?xf32>

        %tl = memref.load %bayer[%y_prev, %x_prev] : memref<?x?xf32>
        %tr = memref.load %bayer[%y_prev, %x_next] : memref<?x?xf32>
        %bl = memref.load %bayer[%y_next, %x_prev] : memref<?x?xf32>
        %br = memref.load %bayer[%y_next, %x_next] : memref<?x?xf32>

        // Compute cross average and diagonal average
        %cross_lr = arith.addf %left, %right : f32
        %cross_tb = arith.addf %top, %bot : f32
        %cross_sum = arith.addf %cross_lr, %cross_tb : f32
        %g_cross = arith.mulf %cross_sum, %c_quarter : f32

        %diag_t = arith.addf %tl, %tr : f32
        %diag_b = arith.addf %bl, %br : f32
        %diag_sum = arith.addf %diag_t, %diag_b : f32
        %diag_avg = arith.mulf %diag_sum, %c_quarter : f32

        %avg_tb = arith.mulf %cross_tb, %c_half : f32
        %avg_lr = arith.mulf %cross_lr, %c_half : f32

        // Select channels based on Bayer RGGB pattern
        // Row 0 (even): R, Gr, R, Gr...
        // Row 1 (odd):  Gb, B, Gb, B...
        %r, %g, %b = scf.if %is_even_row -> (f32, f32, f32) {
          %r_row, %g_row, %b_row = scf.if %is_even_col -> (f32, f32, f32) {
            // Even Row, Even Col: Red Pixel
            scf.yield %center, %g_cross, %diag_avg : f32, f32, f32
          } else {
            // Even Row, Odd Col: Green Pixel (Gr)
            scf.yield %avg_lr, %center, %avg_tb : f32, f32, f32
          }
          scf.yield %r_row, %g_row, %b_row : f32, f32, f32
        } else {
          %r_row, %g_row, %b_row = scf.if %is_even_col -> (f32, f32, f32) {
            // Odd Row, Even Col: Green Pixel (Gb)
            scf.yield %avg_tb, %center, %avg_lr : f32, f32, f32
          } else {
            // Odd Row, Odd Col: Blue Pixel
            scf.yield %diag_avg, %g_cross, %center : f32, f32, f32
          }
          scf.yield %r_row, %g_row, %b_row : f32, f32, f32
        }

        memref.store %r, %rgb[%y, %x, %ch_r] : memref<?x?x3xf32>
        memref.store %g, %rgb[%y, %x, %ch_g] : memref<?x?x3xf32>
        memref.store %b, %rgb[%y, %x, %ch_b] : memref<?x?x3xf32>
      }
    }
    return
  }
}
