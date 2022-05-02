// Multiscale Pyramid Decomposition: 5 Octaves on 1440x2560
module @pyramid_5levels_1440x2560 {
  func.func @pyr_decompose_level(
      %fine: memref<1440x2560xf32>,
      %coarse: memref<720x1280xf32>,
      %laplacian: memref<1440x2560xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %dh = arith.constant 720 : index
    %dw = arith.constant 1280 : index

    // Downsampling decimation pass
    scf.for %oy = %c0 to %dh step %c1 {
      %iy = arith.muli %oy, %c2 : index
      scf.for %ox = %c0 to %dw step %c1 {
        %ix = arith.muli %ox, %c2 : index
        %val = memref.load %fine[%iy, %ix] : memref<1440x2560xf32>
        memref.store %val, %coarse[%oy, %ox] : memref<720x1280xf32>
      }
    }

    // High-frequency bandpass difference pass
    %fh = arith.constant 1440 : index
    %fw = arith.constant 2560 : index
    scf.for %y = %c0 to %fh step %c1 {
      %cy = arith.divui %y, %c2 : index
      scf.for %x = %c0 to %fw step %c1 {
        %cx = arith.divui %x, %c2 : index
        %f_val = memref.load %fine[%y, %x] : memref<1440x2560xf32>
        %c_val = memref.load %coarse[%cy, %cx] : memref<720x1280xf32>
        %diff = arith.subf %f_val, %c_val : f32
        memref.store %diff, %laplacian[%y, %x] : memref<1440x2560xf32>
      }
    }
    return
  }
}
