"""Prompt templates

Adapted from the RelRepair paper for Python programs.
"""

# BaseRepair

BASE_SYSTEM = "You are an expert in Python program repair."

BASE_REPAIR_PROMPT = """\
You are an expert in Python program repair.

## Buggy Function
The following function contains a bug at the indicated location.

File: {file_path}
Fault location: line {fault_line}

```python
{buggy_function}
```

## Error Message
{error_message}

## Failing Test Output
```
{test_output}
```

## Task
Fix the bug in the function above. Return ONLY the complete corrected function, \
enclosed in a single ```python code block. Do not include any explanation.
"""

# ConDefects script-level repair
SCRIPT_REPAIR_PROMPT = """\
You are an expert in Python program repair.

## Buggy Program
The following Python script contains a bug at the indicated line.

File: {task_id}
Fault location: line {fault_line}

```python
{buggy_code}
```

## Task
Fix the bug in the script above. Return ONLY the complete corrected script, \
enclosed in a single ```python code block. Do not include any explanation.
"""
