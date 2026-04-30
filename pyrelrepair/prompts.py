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

# SigRepair: Query Rewriting

SIG_QUERY_REWRITE_PROMPT = """\
You are an expert in Python program repair.

Given the following buggy function, generate two likely root causes of the bug \
and five function names from the codebase that might be relevant to the repair.

## Buggy Function
```python
{buggy_function}
```

Fault location: line {fault_line}

Respond in exactly this format (one item per line, no extra text):
ROOT_CAUSE_1: <first root cause>
ROOT_CAUSE_2: <second root cause>
FUNCTIONS: <func1>, <func2>, <func3>, <func4>, <func5>
"""

# SigRepair: Patch Generation

SIG_REPAIR_PROMPT = """\
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

## Relevant Function Signatures
The following function signatures from the same project may be useful for the repair:

{signatures}

## Task
Using the relevant function signatures above as context, fix the bug. \
Return ONLY the complete corrected function enclosed in a single \
```python code block. Do not include any explanation.
"""


def format_signatures_block(signatures: list[dict]) -> str:
    """Format retrieved signatures into a numbered prompt block."""
    lines = []
    for i, item in enumerate(signatures, 1):
        lines.append(f"{i}. {item['signature']}")
        doc = (item.get("docstring") or "").strip()
        if doc:
            first_line = doc.splitlines()[0]
            lines.append(f"   {first_line}")
    return "\n".join(lines)