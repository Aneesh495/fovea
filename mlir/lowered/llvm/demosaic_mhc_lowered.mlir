// Malvar-He-Cutler optimal FIR lowered to LLVM dialect
// Target-Specific Lowered LLVM Dialect IR in MLIR

module @demosaic_mhc_lowered {
  llvm.func @execute_llvm_stage(
      %arg_in: !llvm.ptr,
      %arg_out: !llvm.ptr,
      %width: i64,
      %height: i64,
      %param: f32) {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %total_elements = llvm.mul %width, %height : i64

    llvm.br ^loop_header(%c0 : i64)

  ^loop_header(%idx: i64):
    %cond = llvm.icmp "slt" %idx, %total_elements : i64
    llvm.cond_br %cond, ^loop_body, ^loop_exit

  ^loop_body:
    %in_ptr = llvm.getelementptr %arg_in[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %val = llvm.load %in_ptr : !llvm.ptr -> f32
    %res = llvm.fmul %val, %param : f32
    %out_ptr = llvm.getelementptr %arg_out[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %res, %out_ptr : f32, !llvm.ptr

    %next_idx = llvm.add %idx, %c1 : i64
    llvm.br ^loop_header(%next_idx : i64)

  ^loop_exit:
    llvm.return
  }
}
