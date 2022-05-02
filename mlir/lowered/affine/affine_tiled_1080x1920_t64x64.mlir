// Polyhedral Affine Tiled Kernel: Resolution 1080x1920, Tile Size 64x64
#map_h = affine_map<(d0) -> (d0 + 64)>
#map_w = affine_map<(d0) -> (d0 + 64)>

module @affine_tiled_1080x1920_t64x64 {
  func.func @tiled_kernel_pass(
      %in: memref<1080x1920x3xf32>,
      %out: memref<1080x1920x3xf32>,
      %scale: f32) {
    affine.for %ty = 0 to 1080 step 64 {
      affine.for %tx = 0 to 1920 step 64 {
        affine.for %y = affine_map<(d0) -> (d0)>(%ty) to min #map_h(%ty) {
          affine.for %x = affine_map<(d0) -> (d0)>(%tx) to min #map_w(%tx) {
            affine.for %c = 0 to 3 {
              %v = affine.load %in[%y, %x, %c] : memref<1080x1920x3xf32>
              %res = arith.mulf %v, %scale : f32
              affine.store %res, %out[%y, %x, %c] : memref<1080x1920x3xf32>
            }
          }
        }
      }
    }
    return
  }
}
