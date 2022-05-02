// ACES Filmic S-Curve Tone Reproduction - Polyhedral Affine Tile 16x64 on 1080x1920
#map_h = affine_map<(d0) -> (d0 + 16)>
#map_w = affine_map<(d0) -> (d0 + 64)>

module @affine_tonemap_aces_1080x1920_t16x64 {
  func.func @execute_tiled(
      %in: memref<1080x1920x3xf32>,
      %out: memref<1080x1920x3xf32>,
      %param: f32) {
    affine.for %ty = 0 to 1080 step 16 {
      affine.for %tx = 0 to 1920 step 64 {
        affine.for %y = affine_map<(d0) -> (d0)>(%ty) to min #map_h(%ty) {
          affine.for %x = affine_map<(d0) -> (d0)>(%tx) to min #map_w(%tx) {
            affine.for %c = 0 to 3 {
              %v = affine.load %in[%y, %x, %c] : memref<1080x1920x3xf32>
              %res = arith.mulf %v, %param : f32
              affine.store %res, %out[%y, %x, %c] : memref<1080x1920x3xf32>
            }
          }
        }
      }
    }
    return
  }
}
