"""Loader for the BugsInPy benchmark dataset.

Reads from data/bugsinpy_checked/ (produced by scripts/bugsinpy_setup.py)
and yields BugInfo objects ready for repair.
"""
from __future__ import annotations

import ast
import json
import logging
from pathlib import Path

from .base_repair import BugInfo

logger = logging.getLogger(__name__)


def _find_function_at_line(
    source: str, fault_line: int
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Return the innermost function node whose body OR decorators contain fault_line."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    candidates: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not hasattr(node, "end_lineno"):
                continue
            # Body range
            if node.lineno <= fault_line <= node.end_lineno:
                candidates.append(node)
                continue
            # Decorator range — lineno of each decorator expression
            for dec in node.decorator_list:
                if hasattr(dec, "end_lineno") and dec.lineno <= fault_line <= dec.end_lineno:
                    candidates.append(node)
                    break

    if not candidates:
        return None
    return max(candidates, key=lambda n: n.lineno)


def _extract_error_message(test_output: str) -> str:
    if not test_output:
        return ""
    for line in test_output.splitlines():
        stripped = line.strip()
        if any(stripped.startswith(p) for p in ("AssertionError", "Error:", "FAILED", "E   ", "assert ")):
            return stripped[:300]
    for line in reversed(test_output.splitlines()):
        if line.strip():
            return line.strip()[:300]
    return test_output[:300]


def _load_single_bug(bug_dir: Path) -> BugInfo | None:
    meta = json.loads((bug_dir / "meta.json").read_text())

    project: str = meta["project"]
    bug_id: int = meta["bug_id"]
    file_path_rel = Path(meta["file_path"])
    fault_line: int = meta["fault_line"]
    test_file_rel: str = meta.get("test_file", "")

    project_dir = bug_dir / "project"
    if not project_dir.exists():
        logger.warning("%s:%s — project dir missing: %s", project, bug_id, project_dir)
        return None

    source_file = project_dir / file_path_rel
    if not source_file.exists():
        logger.warning("%s:%s — source file missing: %s", project, bug_id, source_file)
        return None

    source = source_file.read_text(encoding="utf-8")

    func_node = _find_function_at_line(source, fault_line)
    if func_node is None:
        logger.warning(
            "%s:%s — no function found containing line %d in %s",
            project, bug_id, fault_line, file_path_rel,
        )
        return None

    start_line = func_node.lineno
    end_line = func_node.end_lineno
    # Clamp fault_line into [start_line, end_line] (handles add-only patches)
    fault_line = max(start_line, min(fault_line, end_line))

    lines = source.splitlines()
    buggy_function = "\n".join(lines[start_line - 1 : end_line])
    if not buggy_function.strip():
        logger.warning("%s:%s — extracted empty function", project, bug_id)
        return None

    test_output_file = bug_dir / "test_output.txt"
    test_output = (
        test_output_file.read_text(encoding="utf-8").strip()
        if test_output_file.exists()
        else ""
    )

    error_message = _extract_error_message(test_output)
    test_file = project_dir / test_file_rel if test_file_rel else None

    try:
        return BugInfo(
            bug_id=f"{project}_{bug_id}",
            file_path=file_path_rel,
            function_name=func_node.name,
            buggy_function=buggy_function,
            fault_line=fault_line,
            start_line=start_line,
            end_line=end_line,
            error_message=error_message,
            test_output=test_output[:2000],
            test_file=test_file,
            project_dir=project_dir,
        )
    except ValueError as e:
        logger.warning("%s:%s — BugInfo validation failed: %s", project, bug_id, e)
        return None


def load_bugs(
    checked_dir: Path,
    max_bugs: int | None = None,
) -> list[BugInfo]:
    """Load BugsInPy bugs from the checked-out directory.

    Args:
        checked_dir: Root of bugsinpy_checked/ (contains <project>/<bug_id>/).
        max_bugs: Stop after loading this many bugs.

    Returns:
        List of BugInfo objects with project_dir set, ready for repair.
    """
    bugs: list[BugInfo] = []

    if not checked_dir.exists():
        logger.warning("bugsinpy_checked dir not found: %s", checked_dir)
        return bugs

    for project_dir in sorted(checked_dir.iterdir()):
        if not project_dir.is_dir():
            continue
        for bug_dir in sorted(
            project_dir.iterdir(),
            key=lambda p: int(p.name) if p.name.isdigit() else 0,
        ):
            if not bug_dir.is_dir() or not (bug_dir / "meta.json").exists():
                continue
            try:
                bug = _load_single_bug(bug_dir)
                if bug is not None:
                    bugs.append(bug)
                    logger.info("Loaded %s (fault_line=%d in %s)", bug.bug_id, bug.fault_line, bug.file_path)
                    if max_bugs and len(bugs) >= max_bugs:
                        return bugs
            except Exception as e:
                logger.warning("Failed to load bug from %s: %s", bug_dir, e)

    return bugs
