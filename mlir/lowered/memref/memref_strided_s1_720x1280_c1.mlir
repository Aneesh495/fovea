// Strided MemRef Subview: Stride 1, Shape 720x1280x1
module @memref_strided_s1_720x1280_c1 {
  func.func @strided_subview_copy(
      %source: memref<720x1280x1xf32>,
      %destination: memref<720x1280x1xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c_stride = arith.constant 1 : index
    %h_dim = arith.constant 720 : index
    %w_dim = arith.constant 1280 : index
    %ch_dim = arith.constant 1 : index

    scf.for %sy = %c0 to %h_dim step %c1 {
      %src_y = arith.muli %sy, %c_stride : index
      scf.for %sx = %c0 to %w_dim step %c1 {
        %src_x = arith.muli %sx, %c_stride : index
        scf.for %c = %c0 to %ch_dim step %c1 {
          %val = memref.load %source[%src_y, %src_x, %c] : memref<720x1280x1xf32>
          memref.store %val, %destination[%sy, %sx, %c] : memref<720x1280x1xf32>
        }
      }
    }
    return
  }
}
