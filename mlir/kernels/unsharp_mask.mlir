// Multi-Scale Unsharp Masking Kernel in MLIR
// Boosts high-frequency spatial gradients with coring thresholds to suppress noise amplification.

module @unsharp_mask {
  func.func @unsharp_mask_2d(
      %input: memref<?x?xf32>,
      %blurred: memref<?x?xf32>,
      %output: memref<?x?xf32>,
      %strength: f32,
      %threshold: f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %f0 = arith.constant 0.000000e+00 : f32
    %f1 = arith.constant 1.000000e+00 : f32

    %height = memref.dim %input, %c0 : memref<?x?xf32>
    %width = memref.dim %input, %c1 : memref<?x?xf32>

    scf.for %y = %c0 to %height step %c1 {
      scf.for %x = %c0 to %width step %c1 {
        %orig = memref.load %input[%y, %x] : memref<?x?xf32>
        %blur = memref.load %blurred[%y, %x] : memref<?x?xf32>

        // High-pass gradient = original - blurred
        %high_pass = arith.subf %orig, %blur : f32
        %abs_hp = math.absf %high_pass : f32

        // Coring threshold check: only boost if |high_pass| >= threshold
        %exceeds_threshold = arith.cmpf oge, %abs_hp, %threshold : f32
        %boosted_delta = arith.mulf %high_pass, %strength : f32
        %zero_delta = arith.constant 0.000000e+00 : f32
        %effective_delta = arith.select %exceeds_threshold, %boosted_delta, %zero_delta : f32

        %sharpened = arith.addf %orig, %effective_delta : f32
        %clamped1 = arith.maximumf %sharpened, %f0 : f32
        %final_val = arith.minimumf %clamped1, %f1 : f32

        memref.store %final_val, %output[%y, %x] : memref<?x?xf32>
      }
    }
    return
  }
}
