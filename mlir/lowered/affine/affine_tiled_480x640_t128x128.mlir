// Polyhedral Affine Tiled Kernel: Resolution 480x640, Tile Size 128x128
#map_h = affine_map<(d0) -> (d0 + 128)>
#map_w = affine_map<(d0) -> (d0 + 128)>

module @affine_tiled_480x640_t128x128 {
  func.func @tiled_kernel_pass(
      %in: memref<480x640x3xf32>,
      %out: memref<480x640x3xf32>,
      %scale: f32) {
    affine.for %ty = 0 to 480 step 128 {
      affine.for %tx = 0 to 640 step 128 {
        affine.for %y = affine_map<(d0) -> (d0)>(%ty) to min #map_h(%ty) {
          affine.for %x = affine_map<(d0) -> (d0)>(%tx) to min #map_w(%tx) {
            affine.for %c = 0 to 3 {
              %v = affine.load %in[%y, %x, %c] : memref<480x640x3xf32>
              %res = arith.mulf %v, %scale : f32
              affine.store %res, %out[%y, %x, %c] : memref<480x640x3xf32>
            }
          }
        }
      }
    }
    return
  }
}
