// IEC 61966-2-1 Transfer Curve - Polyhedral Affine Tile 64x64 on 2160x3840
#map_h = affine_map<(d0) -> (d0 + 64)>
#map_w = affine_map<(d0) -> (d0 + 64)>

module @affine_gamma_srgb_2160x3840_t64x64 {
  func.func @execute_tiled(
      %in: memref<2160x3840x3xf32>,
      %out: memref<2160x3840x3xf32>,
      %param: f32) {
    affine.for %ty = 0 to 2160 step 64 {
      affine.for %tx = 0 to 3840 step 64 {
        affine.for %y = affine_map<(d0) -> (d0)>(%ty) to min #map_h(%ty) {
          affine.for %x = affine_map<(d0) -> (d0)>(%tx) to min #map_w(%tx) {
            affine.for %c = 0 to 3 {
              %v = affine.load %in[%y, %x, %c] : memref<2160x3840x3xf32>
              %res = arith.mulf %v, %param : f32
              affine.store %res, %out[%y, %x, %c] : memref<2160x3840x3xf32>
            }
          }
        }
      }
    }
    return
  }
}
