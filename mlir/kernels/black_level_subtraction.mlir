// Black Level Subtraction and Pedestal Compensation Kernel in MLIR
// Linearizes raw sensor measurements by subtracting the dark current pedestal
// and normalizing by the usable dynamic range [black_level, saturation_limit].

module @black_level_subtraction {
  func.func @black_level_2d(
      %raw: memref<?x?xf32>,
      %out: memref<?x?xf32>,
      %offset: f32,
      %inv_scale: f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %f0 = arith.constant 0.000000e+00 : f32
    %f1 = arith.constant 1.000000e+00 : f32

    %dim0 = memref.dim %raw, %c0 : memref<?x?xf32>
    %dim1 = memref.dim %raw, %c1 : memref<?x?xf32>

    scf.for %i = %c0 to %dim0 step %c1 {
      scf.for %j = %c0 to %dim1 step %c1 {
        %val = memref.load %raw[%i, %j] : memref<?x?xf32>
        %sub = arith.subf %val, %offset : f32
        %scaled = arith.mulf %sub, %inv_scale : f32
        %clamped_min = arith.maximumf %scaled, %f0 : f32
        %clamped = arith.minimumf %clamped_min, %f1 : f32
        memref.store %clamped, %out[%i, %j] : memref<?x?xf32>
      }
    }
    return
  }
}
