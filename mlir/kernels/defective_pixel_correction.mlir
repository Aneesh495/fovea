// Defective Pixel Correction (DPC) Kernel in MLIR
// Detects and replaces stuck hot and dead pixels using a cross-neighborhood threshold.

module @defective_pixel_correction {
  func.func @defective_pixel_2d(
      %raw: memref<?x?xf32>,
      %out: memref<?x?xf32>,
      %threshold: f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %f0 = arith.constant 0.000000e+00 : f32
    %c_quarter = arith.constant 0.250000e+00 : f32

    %dim0 = memref.dim %raw, %c0 : memref<?x?xf32>
    %dim1 = memref.dim %raw, %c1 : memref<?x?xf32>

    %dim0_sub2 = arith.subi %dim0, %c2 : index
    %dim1_sub2 = arith.subi %dim1, %c2 : index

    // Copy boundaries directly
    scf.for %i = %c0 to %dim0 step %c1 {
      scf.for %j = %c0 to %dim1 step %c1 {
        %val = memref.load %raw[%i, %j] : memref<?x?xf32>
        memref.store %val, %out[%i, %j] : memref<?x?xf32>
      }
    }

    // Process internal pixels with 2-step same-color cross neighbors
    scf.for %i = %c2 to %dim0_sub2 step %c1 {
      %i_prev2 = arith.subi %i, %c2 : index
      %i_next2 = arith.addi %i, %c2 : index

      scf.for %j = %c2 to %dim1_sub2 step %c1 {
        %j_prev2 = arith.subi %j, %c2 : index
        %j_next2 = arith.addi %j, %c2 : index

        %val = memref.load %raw[%i, %j] : memref<?x?xf32>
        %top = memref.load %raw[%i_prev2, %j] : memref<?x?xf32>
        %bot = memref.load %raw[%i_next2, %j] : memref<?x?xf32>
        %left = memref.load %raw[%i, %j_prev2] : memref<?x?xf32>
        %right = memref.load %raw[%i, %j_next2] : memref<?x?xf32>

        %s1 = arith.addf %top, %bot : f32
        %s2 = arith.addf %left, %right : f32
        %sum = arith.addf %s1, %s2 : f32
        %avg = arith.mulf %sum, %c_quarter : f32

        %diff = arith.subf %val, %avg : f32
        %abs_diff = math.absf %diff : f32

        %is_defective = arith.cmpf ogt, %abs_diff, %threshold : f32
        %repaired = arith.select %is_defective, %avg, %val : f32

        memref.store %repaired, %out[%i, %j] : memref<?x?xf32>
      }
    }
    return
  }
}
