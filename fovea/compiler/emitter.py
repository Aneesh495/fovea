"""
Text emitter for serializing MLIR AST to valid MLIR syntax.
"""
from typing import Any, List
from .ir import Module, Function, Block, Operation, Value, Region

class MLIREmitter:
    """Formats MLIR AST into valid canonical MLIR text."""

    def __init__(self, indent_size: int = 2):
        self.indent_size = indent_size
        self._indent_level = 0
        self._lines: List[str] = []

    def emit_module(self, module: Module) -> str:
        self._lines = []
        self._indent_level = 0
        
        mod_header = f"module @{module.name} {{" if module.name else "module {"
        self._emit_line(mod_header)
        self._indent_level += 1

        for op in module.operations:
            self._emit_operation(op)

        for fn in module.functions:
            self._emit_function(fn)

        self._indent_level -= 1
        self._emit_line("}")
        return "\n".join(self._lines)

    def _emit_line(self, line: str):
        indent = " " * (self._indent_level * self.indent_size)
        self._lines.append(f"{indent}{line}" if line else "")

    def _format_attr(self, val: Any) -> str:
        if isinstance(val, float):
            return f"{val:.6e} : f32" if abs(val) < 1e-4 or abs(val) > 1e4 else f"{val} : f32"
        elif isinstance(val, int):
            return f"{val} : i32"
        elif isinstance(val, str):
            return f'"{val}"'
        elif isinstance(val, list):
            formatted_items = [self._format_attr(x) for x in val]
            return f"[{', '.join(formatted_items)}]"
        return str(val)

    def _emit_function(self, fn: Function):
        args_str = ", ".join(f"%arg{i}: {t.to_mlir()}" for i, t in enumerate(fn.arg_types))
        if len(fn.ret_types) == 0:
            ret_str = ""
        elif len(fn.ret_types) == 1:
            ret_str = f" -> {fn.ret_types[0].to_mlir()}"
        else:
            ret_str = f" -> ({', '.join(t.to_mlir() for t in fn.ret_types)})"

        self._emit_line(f"func.func @{fn.name}({args_str}){ret_str} {{")
        self._indent_level += 1

        for block in fn.region.blocks:
            self._emit_block(block)

        self._indent_level -= 1
        self._emit_line("}")

    def _emit_block(self, block: Block):
        if block.label:
            self._emit_line(f"^{block.label}:")
        for op in block.operations:
            self._emit_operation(op)

    def _emit_operation(self, op: Operation):
        # Format results
        res_str = ""
        if op.results:
            res_str = ", ".join(f"%{r.name}" for r in op.results) + " = "

        # Format operands
        ops_str = ", ".join(f"%{o.name}" for o in op.operands)

        # Format attributes
        attrs_str = ""
        if op.attributes:
            attr_pairs = [f'{k} = {self._format_attr(v)}' for k, v in op.attributes.items()]
            attrs_str = f" {{{', '.join(attr_pairs)}}}"

        # Format type signature
        type_str = ""
        if op.results:
            type_str = f" : {', '.join(r.type.to_mlir() for r in op.results)}"
        elif op.operands:
            type_str = f" : {', '.join(o.type.to_mlir() for o in op.operands)}"

        self._emit_line(f"{res_str}{op.name}({ops_str}){attrs_str}{type_str}")
