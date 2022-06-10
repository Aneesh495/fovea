# Fovea API Reference and CLI Manual

This document details the public Python APIs, data structures, and command-line interfaces provided by the Fovea compiler and image signal processing framework.

---

## 1. CLI Usage (`fovea` / `python3 -m fovea.cli.main`)

Fovea provides a command-line interface for running ISP demonstrations, compiling synthetic targets, and inspecting generated MLIR representations.

### Subcommands

#### `fovea demo`
Executes an end-to-end ISP pipeline pass on a synthetic Macbeth 24-patch ColorChecker chart, generating raw Bayer, demosaiced, and tone-mapped image files.

```bash
# Execute standard demo with default 720x480 resolution
python3 -m fovea.cli.main demo

# Execute demo with custom resolution and output directory
python3 -m fovea.cli.main demo --width 1280 --height 720 --out-dir ./artifacts
```

#### `fovea inspect`
Emits canonical MLIR intermediate representations for any supported kernel or pipeline stage across dialect levels.

```bash
# Inspect Malvar-He-Cutler demosaicing in high-level Fovea dialect
python3 -m fovea.cli.main inspect --kernel demosaic_mhc --dialect fovea

# Inspect tiled polyhedral loop nest in Affine dialect
python3 -m fovea.cli.main inspect --kernel demosaic_mhc --dialect affine

# Inspect vectorized loop kernels in Vector dialect
python3 -m fovea.cli.main inspect --kernel tone_map_aces --dialect vector

# Inspect low-level hardware representation in LLVM dialect
python3 -m fovea.cli.main inspect --kernel color_correction --dialect llvm
```

---

## 2. Python Core API

### `fovea.isp.pipeline.ISPPipeline`
Modular orchestrator for end-to-end raw image signal processing.

```python
from fovea.isp.pipeline import ISPPipeline, ISPPipelineConfig
from fovea.io.raw import RawBayerImage

# Configure pipeline parameters
config = ISPPipelineConfig(
    demosaic_algorithm="malvar_he_cutler",
    enable_lens_shading=True,
    enable_dpc=True,
    enable_denoise=True,
    enable_sharpen=True,
    tonemap_operator="aces"
)

pipeline = ISPPipeline(config)
rgb_result = pipeline.process(raw_bayer_image)
```

#### Methods
- `process(raw_image: RawBayerImage) -> List[float]`: Executes the configured ISP pipeline stages and returns a normalized float RGB image in $[0.0, 1.0]$.
- `process_burst(raw_images: List[RawBayerImage]) -> List[float]`: Executes multi-frame burst fusion using Mertens exposure fusion.

---

### `fovea.compiler.jit.JITEngine`
Native just-in-time compilation engine translating MLIR modules into dynamic libraries and invoking them via ctypes.

```python
from fovea.compiler.jit import JITEngine

jit = JITEngine()
# Compile textual MLIR string into native dylib
dylib_path = jit.compile_module(mlir_text, entry_point="fovea_demosaic")
# Invoke native entry point
jit.invoke(dylib_path, "fovea_demosaic", args=[raw_ptr, out_ptr, 1920, 1080])
```

---

### `fovea.compiler.pass_manager.PassManager`
Automated driver for `mlir-opt` optimization pipelines.

```python
from fovea.compiler.pass_manager import PassManager

pm = PassManager()
pm.add_canonicalize()
pm.add_cse()
pm.add_affine_loop_tile(tile_size=32)
pm.add_super_vectorize(vector_size=8)
pm.add_lowering_to_llvm()

optimized_mlir = pm.run(raw_mlir_text)
```

---

### `fovea.io.raw.RawBayerImage`
Binary reader and writer for packed and unpacked Bayer mosaic sensor dumps with 32-byte standardized binary headers.

```python
from fovea.io.raw import RawBayerImage

# Create in-memory raw frame
raw = RawBayerImage(width=1920, height=1080, cfa_pattern="RGGB", bit_depth=12)
raw.data = [...]

# Save to disk with 32-byte binary header
raw.save("frame_001.raw")

# Load from disk
loaded = RawBayerImage.load("frame_001.raw")
```

---

### `fovea.io.pnm.PNMImage`
High-speed binary reader and writer for Netpbm formats:
- **P6 PPM**: 24-bit RGB binary format.
- **PFM**: 32-bit single-precision floating-point format for high dynamic range imagery.

```python
from fovea.io.pnm import write_ppm_p6, write_pfm, read_ppm_p6

# Write 8-bit display image
write_ppm_p6("output.ppm", width=1920, height=1080, rgb_data=rgb_floats)

# Write 32-bit floating-point HDR image
write_pfm("output.pfm", width=1920, height=1080, rgb_data=hdr_floats)
```
