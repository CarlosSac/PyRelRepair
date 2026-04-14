"""Loader for the ConDefects Python benchmark dataset.

Iterates over Code/{task}/Python/{submission_id}/ directories and yields
ConDefectsBug objects ready for repair.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConDefectsBug:
    submission_id: str
    task_id: str
    fault_line: int
    buggy_code: str
    correct_code: str
    buggy_file: Path


def load_bugs(condefects_dir: Path, max_bugs: int | None = None) -> list[ConDefectsBug]:
    """Load Python bugs from the ConDefects dataset.

    Args:
        condefects_dir: Root of the cloned ConDefects repo (contains Code/).
        max_bugs: If set, stop after loading this many bugs.

    Returns:
        List of ConDefectsBug objects.
    """
    bugs = []
    code_dir = condefects_dir / "Code"

    for task_dir in sorted(code_dir.iterdir()):
        python_dir = task_dir / "Python"
        if not python_dir.is_dir():
            continue

        for submission_dir in sorted(python_dir.iterdir()):
            if not submission_dir.is_dir():
                continue

            faulty = submission_dir / "faultyVersion.py"
            correct = submission_dir / "correctVersion.py"
            fault_loc = submission_dir / "faultLocation.txt"

            if not (faulty.exists() and correct.exists() and fault_loc.exists()):
                continue

            try:
                fault_line = int(fault_loc.read_text(encoding="utf-8").strip())
                buggy_code = faulty.read_text(encoding="utf-8")
                correct_code = correct.read_text(encoding="utf-8")
            except (ValueError, OSError):
                continue

            bugs.append(ConDefectsBug(
                submission_id=submission_dir.name,
                task_id=task_dir.name,
                fault_line=fault_line,
                buggy_code=buggy_code,
                correct_code=correct_code,
                buggy_file=faulty,
            ))

            if max_bugs and len(bugs) >= max_bugs:
                return bugs

    return bugs
