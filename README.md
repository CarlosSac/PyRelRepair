# PyRelRepair

A Python adaptation of **[RelRepair](https://arxiv.org/abs/2509.16701)**, a retrieval-augmented approach to automated program repair. Implements two repair stages, **BaseRepair** (no retrieval baseline) and **SigRepair** (signature-based RAG), evaluated on the **BugsInPy** benchmark. Runs entirely locally via Ollama; no cloud API required.

## Implemented Stages

| Stage          | Strategy                                                 | Status  |
| -------------- | -------------------------------------------------------- | ------- |
| **BaseRepair** | Direct LLM repair with no retrieval                      | Done    |
| **SigRepair**  | Query rewriting + signature retrieval (nomic-embed-text) | Done    |
| **SnipRepair** | Snippet retrieval over full code bodies                  | Planned |

## Requirements

- Python **3.9** (required for test suite compatibility; see Setup below)
- [Ollama](https://ollama.ai/) running locally

### Suggested Ollama models

```bash
ollama pull qwen3-coder:30b      # repair (LLM)
ollama pull nomic-embed-text     # embeddings (SigRepair)
```

The model can be changed in `pyrelrepair/config.py` (`ollama_model` field).

## Setup

Python 3.9 is required because some BugsInPy projects (e.g. tqdm) use `nose` 1.3.7, which relies on the `imp` module removed in Python 3.12.

```bash
# Install Python 3.9 via pyenv if needed
pyenv install 3.9.18
pyenv local 3.9.18

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Datasets

### BugsInPy (primary benchmark)

BugsInPy contains real bugs from popular Python projects (black, tqdm, pandas, keras, etc.). The benchmark only ships metadata; the setup script clones each project at its buggy commit.

First, clone the BugsInPy metadata repository:

```bash
git clone https://github.com/soarsmu/BugsInPy data/bugsinpy
```

Then set up the bugs you want to evaluate:

```bash
# Set up specific bugs
python scripts/bugsinpy_setup.py black:1 black:2 black:3 tqdm:1 tqdm:2 tqdm:3

# Set up all bugs in a project
python scripts/bugsinpy_setup.py --all black
```

This creates `data/bugsinpy_checked/<project>/<id>/` with:

- `project/`: full source tree at the buggy commit
- `meta.json`: fault line, test file, error message
- `test_output.txt`: captured failing test output

### ConDefects (sanity check / model comparison)

Single-file competitive programming bugs. Useful for fast BaseRepair benchmarking across models; SigRepair is not a good fit because ConDefects provides only single files rather than full project codebases to retrieve from.

```bash
git clone https://github.com/appmlk/ConDefects data/condefects
# For stdin/stdout test cases, also download and extract Test.zip from the ConDefects repo
```

## Running Repairs

### BaseRepair on BugsInPy

```bash
python scripts/run_baserepair.py --n 6
```

| Flag                | Default                 | Description                                         |
| ------------------- | ----------------------- | --------------------------------------------------- |
| `--n N`             | 6                       | Number of bugs to evaluate                          |
| `--data PATH`       | `data/bugsinpy_checked` | Path to preprocessed bugs                           |
| `--model M1 M2 ...` | config default          | One or more Ollama models (prints comparison table) |
| `--debug`           | off                     | Save full prompts and responses to `debug.log`      |
| `--verbose / -v`    | off                     | Enable DEBUG-level console output                   |

### SigRepair on BugsInPy

```bash
python scripts/run_sigrepair.py --n 6
```

Same flags as BaseRepair. Requires `nomic-embed-text` to be available in Ollama; the script checks this on startup.

### Full Pipeline (BaseRepair → SigRepair)

```bash
python scripts/run_pipeline.py --n 6
```

Runs BaseRepair first; escalates to SigRepair only for bugs BaseRepair did not fix.

### BaseRepair on ConDefects

```bash
python scripts/run_condefects.py --base-only --n 20

# Compare multiple models
python scripts/run_condefects.py --base-only --n 20 --model qwen3-coder:30b qwen2.5-coder:3b
```

### Output

Results are saved to `results/<stage>_<model>_<timestamp>/`:

- `<bug_id>.json`: patch candidates, test pass/fail counts, token usage
- `summary.json`: aggregate repair rate and token totals
- `debug.log`: full prompt/response log (with `--debug`)

## Configuration

Key settings in `pyrelrepair/config.py`:

| Field                | Default            | Description                          |
| -------------------- | ------------------ | ------------------------------------ |
| `ollama_model`       | `qwen3-coder:30b`  | LLM used for repair                  |
| `embed_model`        | `nomic-embed-text` | Embedding model for SigRepair        |
| `llm_timeout`        | `300`              | Seconds per LLM request              |
| `base_num_patches`   | `1`                | Patches generated per BaseRepair run |
| `sig_num_iterations` | `3`                | SigRepair iterations (paper uses 20) |
| `sig_top_k`          | `25`               | Signatures retrieved per iteration   |

Set `sig_num_iterations = 20` for final evaluation runs.

## How SigRepair Works

Each iteration:

1. **Query rewrite**: the LLM generates 2 root causes and 5 candidate function names from the buggy function
2. **Retrieval**: root causes + function names are embedded with `nomic-embed-text`; top-25 most similar function signatures from the project are retrieved by cosine similarity
3. **Patch generation**: same prompt as BaseRepair, augmented with the retrieved signatures
4. **Validation**: patch is tested with `pytest`; pipeline stops immediately if it passes
5. **Deduplication**: patches identical to ones already tested are skipped

## References

- Liu, et al. (2025). RelRepair. arXiv:2509.16701. https://arxiv.org/abs/2509.16701
- Widyasari, et al. (2020). BugsInPy. FSE 2020.
