# PyRelRepair — Stage 1: Core Infrastructure & Base Repair

A Python adaptation of **RelRepair** (Liu et al., 2025), a retrieval-augmented approach to automated program repair. At this stage the project has the foundational infrastructure and implements the first repair strategy: **BaseRepair**.

## Requirements

- Python 3.12+
- [Ollama](https://ollama.ai/) running locally with a code model pulled

```bash
pip install requests numpy pytest
```

## Usage example for BaseRepair

```python
from pathlib import Path
from pyrelrepair.config import Config
from pyrelrepair.base_repair import BugInfo, base_repair

bug = BugInfo(
    bug_id="example_001",
    file_path=Path("src/module.py"),
    function_name="compute_total",
    buggy_function="def compute_total(items):\n    return sum(items) - 1",
    fault_line=2,
    start_line=1,
    end_line=2,
    error_message="AssertionError: expected 42, got 41",
    test_output="FAILED tests/test_module.py::test_compute_total",
    project_dir=Path("."),
    test_file=Path("tests/test_module.py"),
)

candidates = base_repair(bug, Config())
for c in candidates:
    print(c.is_valid, c.patch_code)
```

Without `project_dir` and `test_file`, patches are syntax-checked only (no pytest).

## Upcoming Stages

- **SigRepair** (Stage 2): query rewriting + SentenceBERT retrieval over function signatures
- **SnipRepair** (Stage 3): CodeBERT retrieval over code snippets with adaptive weight tuning
- **Pipeline**: sequential orchestrator with early exit on first passing patch
