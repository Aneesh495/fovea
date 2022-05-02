// Chromatic Adaptation Gain Map - Polyhedral Affine Tile 32x32 on 720x1280
#map_h = affine_map<(d0) -> (d0 + 32)>
#map_w = affine_map<(d0) -> (d0 + 32)>

module @affine_white_balance_720x1280_t32x32 {
  func.func @execute_tiled(
      %in: memref<720x1280x3xf32>,
      %out: memref<720x1280x3xf32>,
      %param: f32) {
    affine.for %ty = 0 to 720 step 32 {
      affine.for %tx = 0 to 1280 step 32 {
        affine.for %y = affine_map<(d0) -> (d0)>(%ty) to min #map_h(%ty) {
          affine.for %x = affine_map<(d0) -> (d0)>(%tx) to min #map_w(%tx) {
            affine.for %c = 0 to 3 {
              %v = affine.load %in[%y, %x, %c] : memref<720x1280x3xf32>
              %res = arith.mulf %v, %param : f32
              affine.store %res, %out[%y, %x, %c] : memref<720x1280x3xf32>
            }
          }
        }
      }
    }
    return
  }
}
