# Fovea: Multi-Level IR Compiler for Computational Photography and Real-Time ISP

[![Build and Verification](https://github.com/Aneesh495/fovea/actions/workflows/ci.yml/badge.svg)](https://github.com/Aneesh495/fovea/actions)
[![License](https://img.shields.io/badge/License-Apache_2.0_with_LLVM_Exceptions-blue.svg)](LICENSE)
[![MLIR](https://img.shields.io/badge/MLIR-LLVM_23-red.svg)](https://mlir.llvm.org/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-green.svg)](https://python.org)
[![Tested Passes](https://img.shields.io/badge/MLIR_Lines-30%2C731_verified-orange.svg)](dialects/)

Fovea is a domain-specific compiler and multi-level intermediate representation (IR) infrastructure for high-throughput computational photography, Bayer demosaicing, and real-time image signal processing (ISP). It defines a formal MLIR dialect (`fovea`), polyhedral lowering pipelines, sliding-window buffer allocation, SIMD vectorization schedules, and a native JIT execution engine.

The repository contains **30,731 lines of raw, verified MLIR code** spanning 539 generated kernels across four progressive lowering tiers, verified against LLVM `mlir-opt`.

---

## Architecture Overview

Traditional imaging pipelines lower camera sensor processing directly into monolithic C loops or uncomposable GPU kernels. This obscures domain-specific geometric invariants, such as periodic Bayer color filter array (CFA) topologies and overlapping 2D stencil memory footprints. 

Fovea bridges high-level photographic specifications with hardware-specialized polyhedral loop nests and SIMD vector units through a five-tier MLIR dialect stack:

```mermaid
flowchart TD
    RawSensor["Raw Bayer Sensor Frame (10/12/14/16-bit)"] --> FrontEnd["Front-End Python AST / JIT Builder"]
    FrontEnd --> FoveaDialect["Fovea High-Level Dialect (`fovea.*`)"]
    
    subgraph DialectLoweringHierarchy ["Dialect Lowering Pipeline"]
        FoveaDialect -->|Op Conversion & Decomposition| StencilDialect["Affine & Linalg On Tensors"]
        StencilDialect -->|Polyhedral Loop Tiling & Skewing| SCFDialect["SCF & MemRef Dialect"]
        SCFDialect -->|Loop Vectorization & Splatting| VectorDialect["Vector Dialect (`vector<4xf32>`, `vector<8xf32>`)"]
        VectorDialect -->|Hardware Target Expansion| LLVMDialect["LLVM Dialect (`llvm.*`)"]
    end
    
    LLVMDialect --> LLVMIR["LLVM Bitcode / Machine IR"]
    LLVMIR --> TargetBinaries["Target Machine Code (AVX-512, Apple Silicon NEON)"]
```

```
==================================================================================================
                                    FOVEA DIALECT LOWERING STACK
==================================================================================================
   [ Tier 1: Domain IR ]        fovea.bayer_demosaic, fovea.joint_bilateral, fovea.aces_tone_map
                                          │
                                          ▼  (Op Conversion & Decomposition)
   [ Tier 2: Polyhedral IR ]    affine.for, affine.load, affine.store, affine.apply
                                          │  (Loop Tiling, Skewing, Sliding Line Buffers)
                                          ▼
   [ Tier 3: Structured Loops ] scf.for, scf.if, memref.load, memref.store
                                          │
                                          ▼  (Super-Vectorization, Channel Splatting)
   [ Tier 4: Vector IR ]        vector.load, vector.store, vector.fma, vector.broadcast
                                          │
                                          ▼  (Target Expansion & ABI Lowering)
   [ Tier 5: Hardware LLVM ]    llvm.intr.fma, llvm.load, llvm.store, llvm.call
==================================================================================================
```

---

## Key Features

1. **Formal MLIR Dialect Definition**:
   - 14 first-class domain operations covering sensor calibration, reconstruction, enhancement, multi-exposure HDR fusion, and tone mapping.
   - Formally specified in TableGen (`dialects/tablegen/FoveaOps.td`) and verified dynamically in MLIR 23 IRDL (`dialects/irdl/fovea.irdl.mlir`).
2. **Polyhedral Loop Tiling and Stencil Fusion**:
   - Tiled spatial iteration spaces configured to fit within L1 data cache capacities (typically 32 KB to 128 KB).
   - Eliminates 66.7% of DRAM round-trip memory traffic by keeping intermediate color transforms resident in vector registers.
3. **Hardware SIMD Vectorization**:
   - Automated conversion from scalar loops into explicit SIMD vector operations (`vector<4xf32>`, `vector<8xf32>`, `vector<16xf32>`).
   - Hardware fused multiply-accumulate (`vector.fma`) for 2D FIR stencil convolutions.
4. **End-to-End Computational Photography Pipeline**:
   - Sensor Black Level Pedestal Subtraction
   - Radial Lens Shading (Vignetting) Correction
   - Dynamic Defective Pixel Correction (DPC)
   - Bayer Demosaicing (Bilinear, 5x5 Malvar-He-Cutler optimal FIR, Directional Gradient-Adaptive)
   - Illuminant Adaptation (von Kries, Bradford, Gray World estimation)
   - 3x3 Sensor Color Correction Matrix (CCM)
   - Joint Bilateral Filtering and O(1) Moving-Box Guided Filtering
   - Unsharp Masking with Coring Noise Suppression
   - Dyadic Binomial Gaussian and Laplacian Pyramids
   - Multi-Exposure Mertens Burst HDR Fusion
   - Tone Reproduction (Reinhard Global Key, Drago Logarithmic, ACES Filmic Curve)
   - Piecewise IEC 61966-2-1 Standard sRGB EOTF Transfer Function
5. **Zero Mandatory External Dependencies**:
   - Runs out-of-the-box on standard Python 3.10+ without requiring third-party wheels for execution.
   - Transparently integrates with LLVM / MLIR installations (`mlir-opt`, `mlir-translate`, `clang`) when present.

---

## 30k+ Lines of Verified MLIR Suite

The repository includes a comprehensive, verified suite of MLIR implementations generated across 539 modules:

```
dialects/
├── irdl/
│   └── fovea.irdl.mlir         # Verified MLIR 23 IRDL dialect definition
├── tablegen/
│   ├── FoveaDialect.td         # TableGen dialect definition
│   └── FoveaOps.td             # TableGen operation and trait definitions
└── generated/                  # 539 verified MLIR files (30,731 total lines)
    ├── demosaic/               # Malvar-He-Cutler and Bilinear stencils
    ├── calibration/            # Black level, vignette, and DPC kernels
    ├── color/                  # White balance, CCM, and sRGB EOTF transforms
    ├── denoise/                # Joint bilateral and guided filter stencils
    ├── hdr/                    # Mertens multi-exposure burst fusion
    └── tonemap/                # Reinhard and ACES filmic curves
```

Every single `.mlir` file is validated for syntax and semantic compliance:

```bash
# Verify entire 30,731-line suite with LLVM mlir-opt
find dialects/generated -name "*.mlir" -exec mlir-opt {} --allow-unregistered-dialect -o /dev/null \;
```

---

## Performance Benchmarks

Recorded on Apple Silicon ARM64 hardware running LLVM 23.1.4:

### 1. Stencil and Color Transform Fusion

| Execution Model | Pipeline Latency | DRAM Traffic / Frame | Bandwidth Consumed | Traffic Reduction | Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Unfused Stages** | 116.76 ms | 5.53 MB (6 passes) | 47.36 MB/s | Baseline (0%) | 1.00x |
| **Fused Pipeline** | 90.36 ms | 1.84 MB (2 passes) | 20.39 MB/s | **66.7% Reduction** | **1.29x** |

### 2. SIMD Vector Scaling (1-Megapixel Buffer)

| Target Configuration | Vector Width | Mean Latency | Scaling Speedup |
| :--- | :--- | :--- | :--- |
| **Scalar Mode** | 1x (scalar f32) | 211.00 ms | 1.00x |
| **Vector 4x** | 4x (`vector<4xf32>`) | 298.71 ms | 0.71x |
| **Vector 8x** | 8x (`vector<8xf32>`) | 272.86 ms | 0.77x |
| **Vector 16x** | 16x (`vector<16xf32>`) | 258.52 ms | 0.82x |

---

## Quickstart

### 1. Clone and Run CLI Demonstration

```bash
git clone https://github.com/Aneesh495/fovea.git
cd fovea

# Run the end-to-end ISP pipeline demonstration
python3 -m fovea.cli.main demo

# Inspect generated MLIR for any kernel
python3 -m fovea.cli.main inspect --kernel demosaic_mhc --dialect affine
```

### 2. Run Example Workflows

The `examples/` directory contains five complete workflows:

```bash
# 01. Complete Raw Bayer to sRGB Display Pipeline
python3 examples/01_raw_to_rgb_quickstart.py

# 02. Multi-Exposure Burst HDR Fusion (Mertens et al.)
python3 examples/02_burst_hdr_exposure_fusion.py

# 03. Inspect MLIR Pass Transformations
python3 examples/03_inspect_mlir_passes.py

# 04. Custom Kernel Polyhedral Lowering and Vectorization
python3 examples/04_custom_kernel_lowering.py

# 05. Real-Time Camera Stream Simulation
python3 examples/05_realtime_stream_simulation.py
```

### 3. Run Test Suite

```bash
# Run unit, structural, integration, regression, and edge case tests
python3 -m unittest discover -s tests -p "test_*.py"

# Run automated performance benchmarks
python3 benchmarks/run_all_benchmarks.py
```

---

## Python API Example

```python
from fovea.io.raw import RawBayerImage
from fovea.io.synthetic import generate_color_checker_chart, bayer_mosaic
from fovea.isp.pipeline import ISPPipeline, ISPPipelineConfig
from fovea.io.pnm import write_ppm_p6

# 1. Generate synthetic 720p Macbeth ColorChecker and Bayer CFA mosaic
chart = generate_color_checker_chart(width=1280, height=720)
mosaic = bayer_mosaic(chart, width=1280, height=720, cfa_pattern="RGGB")

# 2. Configure ISP pipeline parameters
config = ISPPipelineConfig(
    demosaic_algorithm="malvar_he_cutler",
    enable_lens_shading=True,
    enable_dpc=True,
    enable_denoise=True,
    enable_sharpen=True,
    tonemap_operator="aces"
)

# 3. Process raw sensor frame into display-ready RGB
pipeline = ISPPipeline(config)
rgb_out = pipeline.process(mosaic)

# 4. Save result as standard 24-bit P6 PPM
write_ppm_p6("output_colorchecker.ppm", width=1280, height=720, rgb_data=rgb_out)
```

---

## Technical Documentation

- [Compiler Architecture and Lowering Hierarchy](docs/architecture.md): Multilevel dialect lowering hierarchy, memory traffic analysis, and sliding window buffers.
- [Dialect Specification](docs/dialect_specification.md): Formal specification of all 14 `fovea` dialect operations, attributes, and traits.
- [ISP Mathematical Formulations](docs/isp_pipeline_math.md): Complete mathematical derivations for demosaicing, filtering, HDR fusion, and tone curves.
- [Polyhedral Lowering and Vectorization](docs/lowering_and_fusion.md): Affine loop tiling, super-vectorization, and LLVM machine code generation.
- [Performance Benchmarks](docs/benchmarks.md): Detailed throughput, latency, cache locality, and scaling evaluations.
- [API Reference and CLI Manual](docs/api_reference.md): Developer guide for public Python APIs and command-line tools.

---

## License

This project is licensed under the Apache License 2.0 with LLVM Exceptions. See [LICENSE](LICENSE) for details.
