"""Shared formatting helpers for prompt construction."""
from __future__ import annotations


def code_with_linenos(code: str) -> str:
    """Return code with left-padded line numbers: '  1 | line'."""
    lines = code.splitlines()
    width = len(str(len(lines)))
    return "\n".join(f"{i + 1:{width}d} | {line}" for i, line in enumerate(lines))


def fault_context(code: str, fault_line: int, radius: int = 4) -> tuple[str, str]:
    """Return (fault_line_content, context_block) for the given 1-based fault line.

    Context block shows `radius` lines before/after with an arrow on the fault line.
    """
    lines = code.splitlines()
    fault_content = lines[fault_line - 1].strip() if 0 < fault_line <= len(lines) else ""
    start = max(0, fault_line - 1 - radius)
    end = min(len(lines), fault_line + radius)
    width = len(str(end))
    context_lines = []
    for i in range(start, end):
        marker = ">>>" if i == fault_line - 1 else "   "
        context_lines.append(f"{marker} {i + 1:{width}d} | {lines[i]}")
    return fault_content, "\n".join(context_lines)
