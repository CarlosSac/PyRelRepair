"""Prompt templates

Adapted from the RelRepair paper for Python programs.
"""

# BaseRepair

BASE_SYSTEM = "You are an expert in Python program repair."

BASE_REPAIR_PROMPT = """\
You are an expert in Python program repair. \
The following function contains exactly one bug.

## Function: {function_name} (in {file_path})

```python
{buggy_function_with_linenos}
```

## Fault Location
The bug is on **line {fault_line_relative}** of the function:

```python
{fault_line_content}
```

Surrounding context:
```
{fault_context}
```

## Error
{error_message}

## Failing Test Output
```
{test_output}
```

## Task
Fix the bug. Return ONLY the complete corrected function \
enclosed in a single ```python code block. Do not include any explanation.
"""

# ConDefects script-level repair
SCRIPT_REPAIR_PROMPT = """\
You are an expert in Python competitive programming. \
The following solution received a Wrong Answer verdict. It contains exactly one bug.

## Program: {task_id}

```python
{buggy_code_with_linenos}
```

## Fault Location
The bug is on **line {fault_line}**:

```python
{fault_line_content}
```

Surrounding context:
```
{fault_context}
```

## Task
Fix the bug. Return ONLY the complete corrected script \
enclosed in a single ```python code block. Do not include any explanation.
"""
