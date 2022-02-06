// O(1) Fast Guided Image Filter Kernel in MLIR
// Solves local linear ridge regression: q_i = a_k * I_i + b_k
// to transfer edge structures while regularizing gradient variance.

module @guided_filter {
  func.func @guided_linear_coeff_2d(
      %guide: memref<?x?xf32>,
      %mean_i: memref<?x?xf32>,
      %mean_p: memref<?x?xf32>,
      %mean_ii: memref<?x?xf32>,
      %mean_ip: memref<?x?xf32>,
      %a_out: memref<?x?xf32>,
      %b_out: memref<?x?xf32>,
      %eps: f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %height = memref.dim %guide, %c0 : memref<?x?xf32>
    %width = memref.dim %guide, %c1 : memref<?x?xf32>

    scf.for %y = %c0 to %height step %c1 {
      scf.for %x = %c0 to %width step %c1 {
        %mi = memref.load %mean_i[%y, %x] : memref<?x?xf32>
        %mp = memref.load %mean_p[%y, %x] : memref<?x?xf32>
        %mii = memref.load %mean_ii[%y, %x] : memref<?x?xf32>
        %mip = memref.load %mean_ip[%y, %x] : memref<?x?xf32>

        // var_i = mean_ii - mean_i * mean_i
        %mi_sq = arith.mulf %mi, %mi : f32
        %var_i = arith.subf %mii, %mi_sq : f32

        // cov_ip = mean_ip - mean_i * mean_p
        %mi_mp = arith.mulf %mi, %mp : f32
        %cov_ip = arith.subf %mip, %mi_mp : f32

        // a = cov_ip / (var_i + eps)
        %denom = arith.addf %var_i, %eps : f32
        %a = arith.divf %cov_ip, %denom : f32

        // b = mean_p - a * mean_i
        %a_mi = arith.mulf %a, %mi : f32
        %b = arith.subf %mp, %a_mi : f32

        memref.store %a, %a_out[%y, %x] : memref<?x?xf32>
        memref.store %b, %b_out[%y, %x] : memref<?x?xf32>
      }
    }
    return
  }

  func.func @guided_linear_eval_2d(
      %guide: memref<?x?xf32>,
      %mean_a: memref<?x?xf32>,
      %mean_b: memref<?x?xf32>,
      %output: memref<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %f0 = arith.constant 0.000000e+00 : f32
    %f1 = arith.constant 1.000000e+00 : f32

    %height = memref.dim %guide, %c0 : memref<?x?xf32>
    %width = memref.dim %guide, %c1 : memref<?x?xf32>

    scf.for %y = %c0 to %height step %c1 {
      scf.for %x = %c0 to %width step %c1 {
        %g = memref.load %guide[%y, %x] : memref<?x?xf32>
        %ma = memref.load %mean_a[%y, %x] : memref<?x?xf32>
        %mb = memref.load %mean_b[%y, %x] : memref<?x?xf32>

        %ma_g = arith.mulf %ma, %g : f32
        %q = arith.addf %ma_g, %mb : f32

        %q_clamp1 = arith.maximumf %q, %f0 : f32
        %q_out = arith.minimumf %q_clamp1, %f1 : f32

        memref.store %q_out, %output[%y, %x] : memref<?x?xf32>
      }
    }
    return
  }
}
