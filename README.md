# PyRelRepair: Base Repair

A Python adaptation of **[RelRepair](https://arxiv.org/abs/2509.16701)**, a retrieval-augmented approach to automated program repair. At this stage the project has the foundational infrastructure and implements the first repair strategy: **BaseRepair**, evaluated on the **ConDefects** Python benchmark.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai/) running locally with a code model pulled

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

| Flag          | Default           | Description                                    |
| ------------- | ----------------- | ---------------------------------------------- |
| `--n`         | 10                | Number of bugs to evaluate                     |
| `--data`      | `data/condefects` | Path to ConDefects repo root                   |
| `--verbose`, `-v` | off           | Log prompts, responses, token counts, per-test results |

### Output

```
=== BaseRepair Results ===
Model          : qwen2.5-coder:32b
Bugs evaluated : 50
Syntax valid   : 50/50 (100.0%)
Tokens (prompt): 125,000
Tokens (output):  32,500
Tokens (total) : 157,500
Bugs with tests: 50/50
All tests pass : X/50 (Y%)
```

Validation uses stdin/stdout comparison against the AtCoder test cases. A patch is considered correct only if it passes **all** test cases for that problem.

## Configuration

Key settings in `pyrelrepair/config.py`:

| Field              | Default                  | Description               |
| ------------------ | ------------------------ | ------------------------- |
| `ollama_model`     | `qwen2.5-coder:32b`      | Ollama model name         |
| `ollama_base_url`  | `http://localhost:11434` | Ollama server URL         |
| `temperature`      | `1.0`                    | Sampling temperature      |
| `llm_timeout`      | `300`                    | Seconds per LLM request   |
| `base_num_patches` | `1`                      | Patches generated per bug |

## Programmatic usage (Python API)

For embedding BaseRepair in your own code rather than using the CLI.

```python
from pathlib import Path
from pyrelrepair.config import Config
from pyrelrepair.base_repair import BugInfo, base_repair

bug = BugInfo(
    bug_id="example_001",
    file_path=Path("src/module.py"),
    function_name="compute_total",
    buggy_function="def compute_total(items):\n    return sum(items) - 1",
    fault_line=2,    # 1-based line of the fault within the function
    start_line=1,    # function's first line in the file
    end_line=2,      # function's last line in the file
    error_message="AssertionError: expected 42, got 41",
    test_output="FAILED tests/test_module.py::test_compute_total",
    project_dir=Path("."),                   # omit for syntax-only validation
    test_file=Path("tests/test_module.py"),  # omit to run all project tests
)

candidates = base_repair(bug, Config())
for c in candidates:
    print(c.patch_code)   # only syntax-valid patches are returned
```

With `project_dir` set, use `c.is_valid` to check whether the patch passed the test suite:

```python
candidates = base_repair(bug, Config())
for c in candidates:
    if c.is_valid:        # True only when all tests pass
        print(c.patch_code)
```

## Upcoming Stages

- **SigRepair:** query rewriting + SentenceBERT retrieval over function signatures
- **SnipRepair:**: CodeBERT retrieval over code snippets with adaptive weight tuning
- **Pipeline**: sequential orchestrator with early exit on first passing patch

## References

- Liu, et al. (2025). RelRepair. arXiv:2509.16701. https://arxiv.org/abs/2509.16701
