cmake_minimum_required(VERSION 3.20)
project(fovea LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optional LLVM / MLIR Integration
find_package(MLIR CONFIG QUIET)

if(MLIR_FOUND)
  message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION}")
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)

  include_directories(${LLVM_INCLUDE_DIRS})
  include_directories(${MLIR_INCLUDE_DIRS})
  include_directories(${CMAKE_CURRENT_SOURCE_DIR})

  # TableGen dialect generation
  set(LLVM_TARGET_DEFINITIONS dialects/tablegen/FoveaOps.td)
  mlir_tablegen(FoveaOps.h.inc -gen-op-decls)
  mlir_tablegen(FoveaOps.cpp.inc -gen-op-defs)
  mlir_tablegen(FoveaDialect.h.inc -gen-dialect-decls)
  mlir_tablegen(FoveaDialect.cpp.inc -gen-dialect-defs)
  add_public_tablegen_target(FoveaIncGen)
else()
  message(STATUS "MLIR C++ development package not found. Standalone Python and IR generation mode enabled.")
endif()

# Test target
enable_testing()
find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_test(
  NAME FoveaUnitTests
  COMMAND ${Python3_EXECUTABLE} -m unittest discover -s ${CMAKE_CURRENT_SOURCE_DIR}/tests -p "test_*.py"
)

add_test(
  NAME FoveaLintCheck
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_no_em_dash.py
)
