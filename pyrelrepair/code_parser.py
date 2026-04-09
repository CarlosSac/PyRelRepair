"""Python AST-based code parsing for function extraction."""
from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FunctionInfo:
    """Extracted information about a Python function."""
    name: str
    signature: str  # def name(params) -> return_type
    docstring: str
    body: str  # full function source
    code: str  # body without comments/docstrings
    comments: str  # inline + block comments extracted from body
    file_path: Path
    start_line: int
    end_line: int
    class_name: str | None = None
    decorators: list[str] = field(default_factory=list)

    @property
    def qualified_name(self) -> str:
        if self.class_name:
            return f"{self.class_name}.{self.name}"
        return self.name

    @property
    def signature_with_doc(self) -> str:
        """Signature concatenated with docstring for SigRepair encoding."""
        if self.docstring:
            return f"{self.signature}\n{self.docstring}"
        return self.signature


def _get_source_segment(source_lines: list[str], node: ast.AST) -> str:
    """Extract the source text for an AST node."""
    start = node.lineno - 1
    end = node.end_lineno
    lines = source_lines[start:end]
    if lines:
        lines = textwrap.dedent("\n".join(lines)).strip().split("\n")
    return "\n".join(lines)


def _extract_signature(node: ast.FunctionDef | ast.AsyncFunctionDef, source_lines: list[str]) -> str:
    """Build the function signature string from AST."""
    # Get the def line(s) up to the colon
    start = node.lineno - 1
    # The body starts at the first statement
    if node.body:
        body_start = node.body[0].lineno - 1
    else:
        body_start = start + 1

    sig_lines = source_lines[start:body_start]
    sig_text = "\n".join(sig_lines).strip()
    # Trim to the colon that starts the body
    colon_idx = sig_text.rfind(":")
    if colon_idx != -1:
        sig_text = sig_text[: colon_idx + 1]
    return sig_text


def _extract_comments(body_source: str) -> str:
    """Extract all comments (inline # and docstrings) from function body."""
    comments = []
    for line in body_source.split("\n"):
        stripped = line.strip()
        # Inline comments
        if "#" in stripped:
            idx = stripped.index("#")
            comment = stripped[idx:]
            comments.append(comment)
    return "\n".join(comments)


def _strip_comments_and_docstrings(body_source: str) -> str:
    """Remove comments and docstrings to get pure code."""
    lines = []
    for line in body_source.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # Remove inline comments (naive but sufficient for most cases)
        if "#" in line:
            code_part = line[: line.index("#")]
            if code_part.strip():
                lines.append(code_part.rstrip())
            continue
        lines.append(line)

    code = "\n".join(lines)

    # Remove docstrings (triple-quoted strings at the start of function body)
    code = re.sub(r'^\s*""".*?"""\s*', "", code, flags=re.DOTALL)
    code = re.sub(r"^\s*'''.*?'''\s*", "", code, flags=re.DOTALL)

    return code.strip()


def extract_functions(source_code: str, file_path: Path) -> list[FunctionInfo]:
    """Parse a Python source file and extract all function definitions."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []

    source_lines = source_code.split("\n")
    functions = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        # Determine class context
        class_name = None
        for parent in ast.walk(tree):
            if isinstance(parent, ast.ClassDef):
                for child in ast.iter_child_nodes(parent):
                    if child is node:
                        class_name = parent.name
                        break

        body_source = _get_source_segment(source_lines, node)
        signature = _extract_signature(node, source_lines)
        docstring = ast.get_docstring(node) or ""
        comments = _extract_comments(body_source)
        code = _strip_comments_and_docstrings(body_source)

        decorators = []
        for dec in node.decorator_list:
            dec_source = _get_source_segment(source_lines, dec)
            decorators.append(f"@{dec_source}")

        functions.append(
            FunctionInfo(
                name=node.name,
                signature=signature,
                docstring=docstring,
                body=body_source,
                code=code,
                comments=comments,
                file_path=file_path,
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                class_name=class_name,
                decorators=decorators,
            )
        )

    return functions


def extract_functions_from_file(file_path: Path) -> list[FunctionInfo]:
    """Extract functions from a Python file."""
    try:
        source = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return extract_functions(source, file_path)


def extract_functions_from_directory(
    directory: Path, exclude_tests: bool = False
) -> list[FunctionInfo]:
    """Recursively extract functions from all Python files in a directory."""
    functions = []
    for py_file in directory.rglob("*.py"):
        if exclude_tests and ("test" in py_file.name.lower() or "test" in str(py_file.parent).lower()):
            continue
        functions.extend(extract_functions_from_file(py_file))
    return functions


def get_variable_types(source_code: str) -> list[str]:
    """Extract user-defined type annotations from a function's source.

    Used by SigRepair to find functions related to types used in the
    buggy function (variable-based function collection).
    """
    types_found = set()
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []

    for node in ast.walk(tree):
        # Type annotations on assignments
        if isinstance(node, ast.AnnAssign) and node.annotation:
            type_str = ast.dump(node.annotation)
            if isinstance(node.annotation, ast.Name):
                types_found.add(node.annotation.id)
            elif isinstance(node.annotation, ast.Attribute):
                types_found.add(ast.unparse(node.annotation))

        # Function argument annotations
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                if arg.annotation and isinstance(arg.annotation, ast.Name):
                    types_found.add(arg.annotation.id)

            # Return annotation
            if node.returns and isinstance(node.returns, ast.Name):
                types_found.add(node.returns.id)

    # Filter out built-in types
    builtins = {"int", "float", "str", "bool", "list", "dict", "set", "tuple", "None", "bytes", "type", "object"}
    return [t for t in types_found if t not in builtins]
