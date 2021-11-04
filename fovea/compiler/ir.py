"""
Abstract Syntax Tree and Intermediate Representation data structures for MLIR.
Models operations, blocks, regions, functions, modules, and type representations.
"""
from typing import List, Dict, Any, Optional, Union

class Type:
    """Base class for all MLIR types."""
    def to_mlir(self) -> str:
        raise NotImplementedError

class F32Type(Type):
    def to_mlir(self) -> str:
        return "f32"

class I32Type(Type):
    def to_mlir(self) -> str:
        return "i32"

class IndexType(Type):
    def to_mlir(self) -> str:
        return "index"

class TensorType(Type):
    """Ranked or unranked tensor type: tensor<?x?xf32> or tensor<1080x1920x3xf32>."""
    def __init__(self, shape: List[Union[int, str]], element_type: Type):
        self.shape = shape
        self.element_type = element_type

    def to_mlir(self) -> str:
        dims = "x".join(str(d) if d != -1 else "?" for d in self.shape)
        return f"tensor<{dims}x{self.element_type.to_mlir()}>"

class MemRefType(Type):
    """MemRef type: memref<?x?xf32> or memref<1080x1920x3xf32>."""
    def __init__(self, shape: List[Union[int, str]], element_type: Type):
        self.shape = shape
        self.element_type = element_type

    def to_mlir(self) -> str:
        dims = "x".join(str(d) if d != -1 else "?" for d in self.shape)
        return f"memref<{dims}x{self.element_type.to_mlir()}>"

class VectorType(Type):
    """Vector type: vector<8xf32> or vector<4x4xf32>."""
    def __init__(self, shape: List[int], element_type: Type):
        self.shape = shape
        self.element_type = element_type

    def to_mlir(self) -> str:
        dims = "x".join(str(d) for d in self.shape)
        return f"vector<{dims}x{self.element_type.to_mlir()}>"

class Value:
    """An SSA value in an MLIR module."""
    def __init__(self, name: str, type_: Type):
        self.name = name
        self.type = type_

    def __repr__(self) -> str:
        return f"%{self.name}"

class Operation:
    """An MLIR operation."""
    def __init__(
        self,
        name: str,
        operands: Optional[List[Value]] = None,
        results: Optional[List[Value]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        regions: Optional[List['Region']] = None
    ):
        self.name = name
        self.operands = operands or []
        self.results = results or []
        self.attributes = attributes or {}
        self.regions = regions or []

class Block:
    """An MLIR basic block containing operations."""
    def __init__(self, label: Optional[str] = None, arguments: Optional[List[Value]] = None):
        self.label = label
        self.arguments = arguments or []
        self.operations: List[Operation] = []

    def add_operation(self, op: Operation):
        self.operations.append(op)

class Region:
    """An MLIR region containing blocks."""
    def __init__(self):
        self.blocks: List[Block] = []

    def add_block(self, block: Block):
        self.blocks.append(block)

class Function:
    """A func.func declaration or definition."""
    def __init__(self, name: str, arg_types: List[Type], ret_types: List[Type]):
        self.name = name
        self.arg_types = arg_types
        self.ret_types = ret_types
        self.region = Region()

class Module:
    """A top-level MLIR module."""
    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.functions: List[Function] = []
        self.operations: List[Operation] = []

    def add_function(self, fn: Function):
        self.functions.append(fn)

    def add_operation(self, op: Operation):
        self.operations.append(op)
