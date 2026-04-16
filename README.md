# PyRelRepair: Base Repair

A Python adaptation of **RelRepair** (Liu et al., 2025), a retrieval-augmented approach to automated program repair. At this stage the project has the foundational infrastructure and implements the first repair strategy: **BaseRepair**, evaluated on the **ConDefects** Python benchmark.

## Requirements

- Python 3.12+
- [Ollama](https://ollama.ai/) running locally with a code model pulled

```bash
pip install requests numpy pytest
```

### Default model

`qwen2.5-coder:32b`

The model can be updated in `pyrelrepair/config.py` by changing the `ollama_model` field

```bash
ollama pull qwen2.5-coder:32b
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## ConDefects dataset

Clone the dataset into `data/condefects`:

```bash
git clone https://github.com/appmlk/ConDefects data/condefects
```

For full stdin/stdout validation, download `Test.zip` from the [ConDefects OneDrive link](https://github.com/appmlk/ConDefects?tab=readme-ov-file#test-cases-download), place it in `data/condefects/`, and extract it:

```bash
cd data/condefects && unzip Test.zip
```

Without `Test/`, the runner falls back to syntax-only validation.

## Running BaseRepair on ConDefects

```bash
python scripts/run_baserepair.py --n 50
```

Options:

| Flag     | Default           | Description                  |
| -------- | ----------------- | ---------------------------- |
| `--n`    | 10                | Number of bugs to evaluate   |
| `--data` | `data/condefects` | Path to ConDefects repo root |

### Output

```
=== BaseRepair Results ===
Model          : qwen2.5-coder:32b
Bugs evaluated : 50
Syntax valid   : 50/50 (100.0%)
Bugs with tests: 50/50
All tests pass : X/50 (Y%)
```

Validation uses stdin/stdout comparison against the AtCoder test cases. A patch is considered correct only if it passes **all** test cases for that problem.

## Usage example for BaseRepair (programmatic)

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

## Configuration

Key settings in `pyrelrepair/config.py`:

| Field              | Default                  | Description               |
| ------------------ | ------------------------ | ------------------------- |
| `ollama_model`     | `qwen2.5-coder:32b`      | Ollama model name         |
| `ollama_base_url`  | `http://localhost:11434` | Ollama server URL         |
| `temperature`      | `1.0`                    | Sampling temperature      |
| `llm_timeout`      | `300`                    | Seconds per LLM request   |
| `base_num_patches` | `1`                      | Patches generated per bug |

## Upcoming Stages

- **SigRepair:** query rewriting + SentenceBERT retrieval over function signatures
- **SnipRepair:**: CodeBERT retrieval over code snippets with adaptive weight tuning
- **Pipeline**: sequential orchestrator with early exit on first passing patch
