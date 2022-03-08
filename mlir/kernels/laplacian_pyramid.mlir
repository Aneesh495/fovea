// Dyadic Laplacian Pyramid Reconstruction Kernel in MLIR
// Computes high-frequency band-pass difference: L_k = G_k - Upsample(G_{k+1}).

module @laplacian_pyramid {
  func.func @laplacian_difference_2d(
      %gaussian: memref<?x?xf32>,
      %upsampled: memref<?x?xf32>,
      %laplacian: memref<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %height = memref.dim %gaussian, %c0 : memref<?x?xf32>
    %width = memref.dim %gaussian, %c1 : memref<?x?xf32>

    scf.for %y = %c0 to %height step %c1 {
      scf.for %x = %c0 to %width step %c1 {
        %g = memref.load %gaussian[%y, %x] : memref<?x?xf32>
        %u = memref.load %upsampled[%y, %x] : memref<?x?xf32>
        %diff = arith.subf %g, %u : f32
        memref.store %diff, %laplacian[%y, %x] : memref<?x?xf32>
      }
    }
    return
  }

  func.func @laplacian_reconstruct_2d(
      %laplacian: memref<?x?xf32>,
      %upsampled: memref<?x?xf32>,
      %reconstructed: memref<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %height = memref.dim %laplacian, %c0 : memref<?x?xf32>
    %width = memref.dim %laplacian, %c1 : memref<?x?xf32>

    scf.for %y = %c0 to %height step %c1 {
      scf.for %x = %c0 to %width step %c1 {
        %l = memref.load %laplacian[%y, %x] : memref<?x?xf32>
        %u = memref.load %upsampled[%y, %x] : memref<?x?xf32>
        %sum = arith.addf %l, %u : f32
        memref.store %sum, %reconstructed[%y, %x] : memref<?x?xf32>
      }
    }
    return
  }
}
