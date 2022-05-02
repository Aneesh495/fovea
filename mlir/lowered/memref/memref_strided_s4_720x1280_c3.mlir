// Strided MemRef Subview: Stride 4, Shape 720x1280x3
module @memref_strided_s4_720x1280_c3 {
  func.func @strided_subview_copy(
      %source: memref<720x1280x3xf32>,
      %destination: memref<180x320x3xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c_stride = arith.constant 4 : index
    %h_dim = arith.constant 180 : index
    %w_dim = arith.constant 320 : index
    %ch_dim = arith.constant 3 : index

    scf.for %sy = %c0 to %h_dim step %c1 {
      %src_y = arith.muli %sy, %c_stride : index
      scf.for %sx = %c0 to %w_dim step %c1 {
        %src_x = arith.muli %sx, %c_stride : index
        scf.for %c = %c0 to %ch_dim step %c1 {
          %val = memref.load %source[%src_y, %src_x, %c] : memref<720x1280x3xf32>
          memref.store %val, %destination[%sy, %sx, %c] : memref<180x320x3xf32>
        }
      }
    }
    return
  }
}
