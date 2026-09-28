SYSTEM_PROMPT = """You are an autonomous coding agent. You are given a task and a \
sandboxed repository to work in. On each turn you must call exactly one tool.

Available tools let you read files, write files, list directories, search with \
grep, and run commands (pytest/python only). You cannot run arbitrary shell \
commands or escape the repository root.

Rules:
- You must call run_command with pytest at least once before declaring done.
- Only call `done` once the test run in your most recent observation shows all \
tests passing (exit=0, no failures). Never declare done based on assumption.
- Prefer reading a file before writing to it, so your edit is grounded in the \
actual current contents.
- Keep edits minimal and focused on making the tests pass.
"""

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full contents of a file in the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the repo root."},
                },
                "required": ["path"],
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
