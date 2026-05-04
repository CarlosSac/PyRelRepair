#!/usr/bin/env python3
"""
Set up BugsInPy bugs for the PyRelRepair pipeline.

Clones each selected bug's project at the buggy commit, installs it,
runs the failing test, and saves metadata + test output to
data/bugsinpy_checked/<project>/<bug_id>/.

Usage:
    python scripts/bugsinpy_setup.py black:1 black:2 tqdm:1 tqdm:2
    python scripts/bugsinpy_setup.py --all black
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

BUGSINPY_DIR = Path("data/bugsinpy")
CHECKED_DIR = Path("data/bugsinpy_checked")
VENV_PYTHON = sys.executable


def _parse_info(text: str) -> dict[str, str]:
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if "=" in line:
            key, _, val = line.partition("=")
            result[key.strip()] = val.strip().strip('"')
    return result


def _parse_patch(patch_text: str) -> tuple[str | None, int | None]:
    """Return (file_path, fault_line) from a git diff.

    fault_line is the 1-based line in the original file of the first removed
    line.  For add-only patches it falls back to the hunk start line.
    """
    file_path: str | None = None
    fault_line: int | None = None
    current_line = 0
    hunk_start: int | None = None
    in_hunk = False

    for line in patch_text.splitlines():
        if line.startswith("--- a/") and line.endswith(".py"):
            file_path = line[6:]
            fault_line = None
            hunk_start = None
            current_line = 0
            in_hunk = False
        elif line.startswith("@@"):
            m = re.match(r"@@ -(\d+)", line)
            if m:
                current_line = int(m.group(1))
                in_hunk = True
                if hunk_start is None:
                    hunk_start = current_line
        elif in_hunk and file_path is not None and fault_line is None:
            if line.startswith("-") and not line.startswith("---"):
                fault_line = current_line
                current_line += 1
            elif line.startswith("+") and not line.startswith("+++"):
                pass  # added lines don't exist in the original file
            else:
                current_line += 1  # context line

    if fault_line is None and hunk_start is not None and file_path is not None:
        fault_line = hunk_start  # add-only patch: use insertion point

    return file_path, fault_line


def _clone_at_commit(github_url: str, commit: str, dest: Path) -> bool:
    if (dest / ".git").exists():
        print(f"  Already cloned: {dest}")
        return True

    dest.mkdir(parents=True, exist_ok=True)
    print(f"  Cloning {github_url} ...")
    r = subprocess.run(
        ["git", "clone", "--quiet", github_url, str(dest)],
        capture_output=True, text=True, timeout=300,
    )
    if r.returncode != 0:
        print(f"  Clone failed: {r.stderr[:300]}")
        return False

    r = subprocess.run(
        ["git", "checkout", commit],
        cwd=str(dest), capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        print(f"  Checkout {commit[:8]} failed: {r.stderr[:200]}")
        return False

    print(f"  Checked out {commit[:8]}")
    return True


def _install_project(project_dir: Path) -> None:
    for marker in ("pyproject.toml", "setup.py", "setup.cfg"):
        if (project_dir / marker).exists():
            print("  Installing project (pip install -e .) ...")
            subprocess.run(
                [VENV_PYTHON, "-m", "pip", "install", "-e", ".", "--quiet"],
                cwd=str(project_dir), capture_output=True, timeout=180,
            )
            return


def _run_test(test_cmd: str, project_dir: Path, timeout: int = 90) -> str:
    cmd = test_cmd.strip()
    python_q = shlex.quote(VENV_PYTHON)
    for prefix in ("python3 ", "python "):
        if cmd.startswith(prefix):
            cmd = python_q + " " + cmd[len(prefix):]
            break

    print(f"  Running: {cmd}")
    try:
        r = subprocess.run(
            cmd, shell=True, cwd=str(project_dir),
            capture_output=True, text=True, timeout=timeout,
        )
        return (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return "TIMEOUT: test exceeded time limit"
    except Exception as e:
        return f"ERROR running test: {e}"


def setup_bug(project: str, bug_id: int) -> bool:
    bug_meta_dir = BUGSINPY_DIR / "projects" / project / "bugs" / str(bug_id)
    project_info_path = BUGSINPY_DIR / "projects" / project / "project.info"

    if not bug_meta_dir.exists():
        print(f"  Not found: {bug_meta_dir}")
        return False

    project_info = _parse_info(project_info_path.read_text())
    bug_info = _parse_info((bug_meta_dir / "bug.info").read_text())
    patch_text = (bug_meta_dir / "bug_patch.txt").read_text()
    test_cmd = (bug_meta_dir / "run_test.sh").read_text().strip()

    github_url = project_info.get("github_url", "")
    buggy_commit = bug_info.get("buggy_commit_id", "")
    test_file = bug_info.get("test_file", "")

    if not github_url or not buggy_commit:
        print("  Missing github_url or buggy_commit_id in project/bug info")
        return False

    file_path, fault_line = _parse_patch(patch_text)
    if file_path is None or fault_line is None:
        print("  Could not extract file_path/fault_line from patch")
        return False

    out_dir = CHECKED_DIR / project / str(bug_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    project_checkout = out_dir / "project"

    if not _clone_at_commit(github_url, buggy_commit, project_checkout):
        return False

    # Run setup.sh (best effort)
    setup_sh = bug_meta_dir / "setup.sh"
    if setup_sh.exists():
        for line in setup_sh.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                subprocess.run(
                    line, shell=True, cwd=str(project_checkout),
                    capture_output=True, timeout=120,
                )

    _install_project(project_checkout)

    test_output = _run_test(test_cmd, project_checkout)
    (out_dir / "test_output.txt").write_text(test_output, encoding="utf-8")

    meta = {
        "project": project,
        "bug_id": bug_id,
        "github_url": github_url,
        "buggy_commit_id": buggy_commit,
        "file_path": file_path,
        "fault_line": fault_line,
        "test_file": test_file,
        "test_cmd": test_cmd,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"  file_path={file_path}  fault_line={fault_line}")
    print(f"  Saved → {out_dir}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up BugsInPy bugs for PyRelRepair")
    parser.add_argument("bugs", nargs="*", help="Bug specs like black:1 tqdm:2")
    parser.add_argument("--all", metavar="PROJECT", help="Set up all bugs in a project")
    args = parser.parse_args()

    targets: list[tuple[str, int]] = []

    if args.all:
        bugs_root = BUGSINPY_DIR / "projects" / args.all / "bugs"
        for b in sorted(bugs_root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 0):
            if b.is_dir():
                targets.append((args.all, int(b.name)))

    for spec in args.bugs:
        if ":" in spec:
            project, _, bid = spec.partition(":")
            targets.append((project, int(bid)))
        else:
            print(f"Skipping invalid spec '{spec}' — expected project:bug_id")

    if not targets:
        parser.print_help()
        sys.exit(1)

    ok = 0
    for project, bug_id in targets:
        print(f"\n=== {project}:{bug_id} ===")
        if setup_bug(project, bug_id):
            ok += 1
        else:
            print("  FAILED")

    print(f"\nDone: {ok}/{len(targets)} bugs ready in {CHECKED_DIR}/")


if __name__ == "__main__":
    main()
