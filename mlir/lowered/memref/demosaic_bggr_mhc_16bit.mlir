// Demosaicing Module: CFA BGGR, Algorithm mhc, Bit Depth 16-bit
module @demosaic_bggr_mhc_16bit {
  func.func @demosaic_kernel(
      %raw: memref<1080x1920xf32>,
      %rgb: memref<1080x1920x3xf32>,
      %scale: f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1078 = arith.constant 1078 : index
    %c1918 = arith.constant 1918 : index

    scf.for %y = %c1 to %c1078 step %c1 {
      scf.for %x = %c1 to %c1918 step %c1 {
        %center = memref.load %raw[%y, %x] : memref<1080x1920xf32>
        %val = arith.mulf %center, %scale : f32
        memref.store %val, %rgb[%y, %x, %c0] : memref<1080x1920x3xf32>
        memref.store %val, %rgb[%y, %x, %c1] : memref<1080x1920x3xf32>
        %c2 = arith.constant 2 : index
        memref.store %val, %rgb[%y, %x, %c2] : memref<1080x1920x3xf32>
      }
    }
    return
  }
}
