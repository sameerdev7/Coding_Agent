SYSTEM_PROMPT = """You are an autonomous coding agent. You are given a task and a \
sandboxed repository to work in. On each turn you must call exactly one tool.

Available tools let you read files, edit or write files, list directories, \
search with grep, and run commands (pytest/python only). You cannot run \
arbitrary shell commands or escape the repository root.

Rules:
- You must call run_command with pytest at least once before declaring done.
- Only call `done` once the test run in your most recent observation shows all \
tests passing (exit=0, no failures). Never declare done based on assumption.
- Test files are read-only. Fix the implementation, never the tests; edits to \
test files are rejected, and `done` is refused if any test file changed.
- Prefer reading a file before changing it, so your edit is grounded in the \
actual current contents.
- Prefer `edit_file` (replace one exact snippet) over `write_file` (rewrite the \
whole file) for small changes.
- Tool output is truncated when long. For large files, read a slice with \
start_line/end_line instead of the whole file.
- Keep edits minimal and focused on making the tests pass.
"""

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file in the repository, optionally only a range of lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the repo root."},
                    "start_line": {"type": "integer", "description": "First line to return (1-indexed). Optional."},
                    "end_line": {"type": "integer", "description": "Last line to return (inclusive). Optional."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Replace exactly one occurrence of old_text with new_text in a file. Fails if old_text "
                "is missing or appears more than once. Prefer this over write_file for small changes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the repo root."},
                    "old_text": {
                        "type": "string",
                        "description": "Exact text to replace (must be unique in the file).",
                    },
                    "new_text": {"type": "string", "description": "Replacement text."},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Overwrite a file with new content. Creates the file if it does not exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the repo root."},
                    "content": {"type": "string", "description": "The full new contents of the file."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List entries in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the repo root. Defaults to '.'."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search for a pattern in the repository (like grep -rn).",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "The pattern to search for."},
                    "path": {"type": "string", "description": "Path relative to the repo root. Defaults to '.'."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run an allowlisted command (pytest, python, or python3) in the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The command and its arguments, e.g. ['pytest', '-q'].",
                    },
                },
                "required": ["args"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Declare the task complete. Only call this after confirming tests pass.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "A short summary of what was fixed."},
                },
                "required": [],
            },
        },
    },
]
