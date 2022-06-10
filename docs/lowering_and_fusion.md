# Polyhedral Lowering, Loop Tiling, and Vectorization

This document details the compiler transformations, pass pipelines, and code generation strategies that Fovea employs to lower high-level domain operations into optimized machine code.

---

## 1. Compilation Pass Pipeline

Fovea drives the LLVM MLIR pass manager (`mlir-opt`) through an orchestrated sequence of polyhedral, loop, and target-lowering passes:

```mermaid
flowchart TD
    A["`fovea` High-Level IR Module"] --> B["`--canonicalize`, `--cse`"]
    B --> C["Dialect Lowering (`fovea` -> `affine` / `scf`)"]
    C --> D["Polyhedral Loop Tiling (`--affine-loop-tile=tile-size=32`)"]
    D --> E["Affine Loop Unrolling & Jamming (`--affine-loop-unroll=unroll-factor=4`)"]
    E --> F["Affine Super Vectorization (`--affine-super-vectorize=virtual-vector-size=8`)"]
    F --> G["Vector to LLVM Conversion (`--convert-vector-to-llvm`)"]
    G --> H["SCF & Control Flow Lowering (`--convert-scf-to-cf`, `--convert-cf-to-llvm`)"]
    H --> I["Func to LLVM Conversion (`--convert-func-to-llvm`)"]
    I --> J["Native Machine Code Translation (`mlir-translate`, Clang `-O3`)"]
```

---

## 2. Polyhedral Loop Tiling for Cache Residency

In stencil processing (e.g. 5x5 Malvar-He-Cutler demosaicing or 5x5 binomial unsharp masking), the naive row-major iteration order sweeps through full image scanlines. For a $3840 \times 2160$ frame, a single scanline spans 15 KB (1-channel 12-bit) to 46 KB (3-channel float32). Accessing 5 adjacent scanlines simultaneously requires 230 KB of cache footprint, exceeding the typical L1 data cache capacity (32 KB to 128 KB) and causing cache thrashing.

### Rectangular Loop Tiling
Fovea applies affine loop tiling across spatial induction variables $(y, x)$:

```mlir
// Tiled 2D Loop Nest: Tile size 32x64
affine.for %y_tile = 0 to 2160 step 32 {
  affine.for %x_tile = 0 to 3840 step 64 {
    affine.for %y = #map_min(%y_tile) to #map_max(%y_tile + 32) {
      affine.for %x = #map_min(%x_tile) to #map_max(%x_tile + 64) {
        // Point computation with guaranteed L1 cache residency
        %elem = affine.load %src[%y, %x] : memref<2160x3840xf32>
        // ...
      }
    }
  }
}
```

Within a $32 \times 64$ tile, the active working set is:

$$
\text{Working Set} = (32 + 2R) \times (64 + 2R) \times \text{sizeof(float)}
$$

For a radius $R = 2$ (5x5 stencil), the tile requires $(36 \times 68 \times 4) = 9.79 \text{ KB}$, easily fitting inside standard L1 data cache memory.

---

## 3. SIMD Super-Vectorization

Fovea leverages the MLIR Vector dialect to generate hardware-agnostic SIMD primitives before mapping to hardware instructions (x86 AVX2/AVX-512, ARM NEON).

### Vectorization Strategy:
1. **Contiguous Stride-1 Dimension**: The innermost spatial dimension $x$ is vectorized into contiguous chunks of vector size $V \in \{4, 8, 16\}$:
   ```mlir
   %v_in = vector.load %mem[%y, %x] : memref<2160x3840xf32>, vector<8xf32>
   ```
2. **Channel-Splatted Arithmetic**: For point-wise color transformations (white balance, color correction matrix), channel gains are broadcast into vector registers:
   ```mlir
   %gain_splat = vector.broadcast %gain_val : f32 to vector<8xf32>
   %v_scaled = arith.mulf %v_in, %gain_splat : vector<8xf32>
   ```
3. **Hardware Fused Multiply-Accumulate (FMA)**: Stencil convolutions are mapped to `vector.fma`:
   ```mlir
   %acc_next = vector.fma %v_sample, %v_coeff, %acc : vector<8xf32>
   ```

---

## 4. Hardware Translation and JIT Execution

The final lowering step targets LLVM IR:

1. **MemRef Descriptor Expansion**: Multi-dimensional `memref` buffers expand into standard LLVM struct descriptors containing:
   - Allocated pointer (`!llvm.ptr`)
   - Aligned pointer (`!llvm.ptr`)
   - Offset (`i64`)
   - Sizes (`array<2 x i64>`)
   - Strides (`array<2 x i64>`)
2. **Dynamic JIT Invocation**: The resulting `.dylib` or `.so` is mapped into process memory. C-compatible ABI function signatures allow execution directly on native memory buffers:
   ```c
   void fovea_pipeline_mhc_aces(
       float* raw_mosaic,
       float* rgb_out,
       int64_t width,
       int64_t height
   );
   ```
   No inter-process communication or file serialization occurs during processing.
