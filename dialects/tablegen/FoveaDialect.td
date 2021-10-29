// Fovea Dialect TableGen Definition
// Defines the high-level computational photography and ISP dialect.

#ifndef FOVEA_DIALECT_TD
#define FOVEA_DIALECT_TD

include "mlir/IR/OpBase.td"

def Fovea_Dialect : Dialect {
  let name = "fovea";
  let summary = "High-performance computational photography and ISP dialect";
  let description = [{
    The `fovea` dialect provides first-class intermediate representation abstractions
    for computational imaging, sensor demosaicing, multi-exposure HDR fusion, and
    end-to-end image signal processor (ISP) pipelines.
  }];
  let cppNamespace = "::mlir::fovea";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif // FOVEA_DIALECT_TD
