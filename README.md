# Coding Agent

[![CI](https://github.com/sameerdev7/Coding_Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/sameerdev7/Coding_Agent/actions/workflows/ci.yml)

An autonomous coding agent that reads a repo, figures out what's broken, edits the code, reruns the test suite, and keeps going until it's actually green — no human in the loop.

I built this to prove out the *mechanism* behind agentic coding tools, not just call one. The orchestration is a hand-rolled [LangGraph](https://github.com/langchain-ai/langgraph) state machine (deliberately zero LangChain agent/chain abstractions), commands run in a locked-down Docker container, the agent can't "fix" a failing test by editing it, and the whole control loop is unit-tested with a scripted fake LLM so the test suite costs nothing and needs no API key.

![Agent fixing a bug in the demo repo, unattended, real terminal output](assets/demo-screenshot.png)

## What it actually does

Given a repo and a task like *"make all tests pass"*, the agent first writes a short plan in one upfront LLM call, then loops:

```
plan_overview (once) → plan → execute_tool → observe → (tests passing and done? stop : keep going)
```

On each turn the LLM sees the conversation so far and must call exactly one tool: `read_file` (optionally a line range), `edit_file` (replace one exact snippet), `write_file`, `list_dir`, `grep`, or `run_command` (allowlisted to `pytest`/`python`). It can only declare `done` after a `pytest` run in its own last observation came back clean, and only if no test file changed — the graph checks both itself and won't take the model's word for it.

If the target repo is its own git repository, the agent works on a fresh branch (never your current one) and only commits if the fix verifies. A failed attempt is left uncommitted on its branch for you to inspect, never silently merged.

## Quickstart

```bash
git clone git@github.com:sameerdev7/Coding_Agent.git
cd Coding_Agent
uv sync
```

Get a free Groq key at [console.groq.com](https://console.groq.com) (API Keys → Create, no card needed), then create a `.env` in the project root:

```
LLM_PROVIDER=groq
GROQ_API_KEY=your-key-here
GEMINI_API_KEY=
```

Run it against the seeded demo bug:

```bash
uv run coding-agent run --repo ./demo_repo_multifile --task "Make all tests in tests/ pass"
```

(`uv run python cli.py ...` works too.) A live trace as it works:

```
🔒 sandbox: docker (no network, read-only rootfs, non-root, no capabilities, resource-limited)
[1] plan     -> run_command({'args': ['pytest', '-q']})
[1] observe  -> exit=1
[2] plan     -> read_file({'end_line': 200, 'path': 'orders.py', 'start_line': 1})
[2] observe  -> from config import STATUS_LABELS
[3] plan     -> read_file({'end_line': 200, 'path': 'config.py', 'start_line': 1})
[3] observe  -> STATUS_LABELS = {
[4] plan     -> edit_file({'new_text': '"shipped": "Shipped",', 'old_text': '"shiped": "Shipped",', 'path': 'config.py'})
[4] observe  -> edited config.py: replaced 1 occurrence
[5] plan     -> run_command({'args': ['pytest', '-q']})
[5] observe  -> exit=0
[6] observe  -> Error: you must call a tool. Use `done` only once tests pass.
[7] plan     -> done({'summary': "Fixed typo in STATUS_LABELS: corrected 'shiped' to 'shipped'"})
[7] observe  -> acknowledged

✅ Task complete in 7 iterations (9,105 tokens). Tests passing.
```

That's a real run, verbatim apart from the plan paragraph printed before step 1 — one of the faster ones (see the reliability table for typical iteration counts). The bug's root cause is in a different file than the failing traceback points to — the agent had to follow an import to find it.

Other modes:

```bash
uv run coding-agent chat --repo ./demo_repo_logic    # follow-up tasks in one session
uv run coding-agent tui  --repo ./demo_repo_logic    # same, as a Textual terminal UI
uv run coding-agent eval --runs 3                    # reliability: N runs per demo repo
```

## Security model

Letting an LLM run code on your machine is the dangerous part, so it's worth being exact about what is and isn't protected.

| Layer | What it does | What it does *not* do |
|---|---|---|
| **File tools** (`read/write/edit/list/grep`) | Every path is resolved and must stay inside the repo root; `..`, absolute paths and symlink escapes are rejected. `grep` can't be tricked with `-`-prefixed patterns. | Run inside the container — they run on the host, so the path jail is what protects them. |
| **`run_command` allowlist** | Only `pytest`, `python`, `python3`; `shell=False`; timeout clamped to 120s. | Limit what those programs *do*: `python` and `pytest` execute arbitrary code, so the allowlist alone is **not** isolation. |
| **Docker sandbox** (default when Docker is available) | Commands run with `--network none`, read-only root filesystem, non-root user, `--cap-drop ALL`, `no-new-privileges`, memory/CPU/PID limits, and only the repo mounted. Tested for real: no network, can't write outside `/workspace`, can't see host files, timeout kills the container. | Defend against a container escape (it's a standard hardened container, not a VM), or give the repo's third-party dependencies to the container — only `pytest` is installed, with no network. |
| **`--sandbox local`** | Allowlist + timeout only. Printed in yellow as `NO isolation` every run. | Anything else. Use it only on repos you trust. |
| **Test protection** | Test files are read-only to `write_file`/`edit_file`; a hash snapshot is taken at the start and `done` is refused if any test file changed — which also catches `python -c "open('tests/...').write(...)"`. | Stop the agent from writing a *bad implementation* that special-cases the tests. |
| **Git safety** | Works on its own branch, commits only on verified success, refuses to touch a dirty repo. | Push anywhere or open PRs. |

`--sandbox auto` (the default) uses Docker if it's reachable and falls back to local with a visible warning; `--sandbox docker` fails loudly instead of falling back. Your repo's contents are sent to the LLM provider (Groq/Gemini) as part of the conversation — don't point this at code you can't share with them.

## Reliability

Measured with `coding-agent eval --runs 3 --sandbox docker` on 2026-10-05: each run starts from a fresh copy of the repo, confirms the baseline really fails, runs the agent, then **re-verifies with pytest itself** rather than trusting the agent's "done". Model: `openai/gpt-oss-120b` on Groq's free tier.

| repo | runs | success | avg iterations (on success) | avg tokens | avg seconds | errors |
|---|---|---|---|---|---|---|
| demo_repo (missing guards) | 3 | 3/3 (100%) | 12.0 | 15,497 | 123.2 | 0 |
| demo_repo_logic (wrong logic) | 3 | 3/3 (100%) | 11.3 | 14,373 | 105.9 | 0 |
| demo_repo_multifile (root cause in another file) | 3 | 3/3 (100%) | 10.7 | 14,006 | 79.1 | 0 |
| demo_repo_git (own git history) | 3 | 3/3 (100%) | 8.7 | 12,867 | 69.1 | 0 |
| **overall** | 12 | **12/12 (100%)** | 10.7 | 14,186 | 94.3 | 0 |

What this does and doesn't show: the loop reliably fixes small, well-specified bugs end to end inside the sandbox. It is 12 runs on 4 tiny bugs with one model, so it says nothing about real-world repos — the demos are deliberately small, and a 100% rate on them is a floor for "does the mechanism work", not a benchmark. Iteration counts (~11) are higher than the 6–8 I saw in early hand-run demos; some of that is run-to-run variance, but I haven't investigated whether history compaction makes the model re-read files it no longer sees in full. Rerun it yourself: `uv run coding-agent eval --runs 3`.

## Production-grade features

- **Isolated execution and test protection** — see the security model above.
- **Context and token control.** Tool output is capped (head + tail kept, since exit codes live at the head and pytest summaries at the tail); the model sees a *compacted* history where older tool results and whole-file writes are elided while the full history is kept in state for transcripts; `read_file` takes line ranges and `edit_file` replaces a single snippet, so the model doesn't have to re-emit a whole file to change one line. Token usage is tracked per run and reported.
- **Git workflow, not raw file overwrites.** Fresh `agent/<task>-<timestamp>` branch, commit only on verified success, real `git diff` shown after every edit. Deterministic — the LLM never decides when to commit. Opt out with `--no-git`.
- **A real planning step**, then a reactive tool loop.
- **Observability.** `--run-log run.jsonl` appends one JSON line per event (start, plan, each step with tool/args/result/tokens, end with success/iterations/tokens/seconds) — an audit trail you can replay or grep.
- **An eval harness** (`agent/evals.py`) that runs each repo N times on a fresh copy, checks the baseline really fails, and **verifies success independently with pytest rather than trusting the agent's own "done"**.
- **Interactive modes.** `chat` and a Textual `tui` (graph runs in a worker thread so the UI never freezes), both built on the same event stream as the CLI.
- **Typed, validated config** (`pydantic-settings`): `LLM_PROVIDER`, `SANDBOX`, `PROTECT_TESTS`, `MAX_ITERATIONS` — a bad value fails fast with a clear error, not silently.
- **Provider adapters with real failure handling.** Groq: retries 429s using the server-suggested delay, turns hallucinated-tool 400s into a correctable observation. Gemini: retries per-minute limits, fails fast on daily quota. Both covered by stubbed-SDK tests.
- **CI**: lint + format (`ruff`), types (`mypy`), tests with an 85% coverage gate, and a Docker job that re-verifies the sandbox's isolation guarantees on every push. A manual workflow runs the live eval.
- **Installable**: `coding-agent` console script via `pyproject.toml`.

## Architecture

```
agent/
  state.py       AgentState — the whole graph is a pure function over this
  graph.py         the LangGraph loop: plan_overview, plan, execute_tool (policy lives here), observe, route
  tools.py         path-jailed file tools + run_command
  sandbox.py       LocalSandbox / DockerSandbox — where run_command actually executes
  integrity.py     test-file protection: path rules + hash snapshots
  context.py       output truncation + history compaction
  events.py        one structured event stream (PlanEvent / StepEvent) consumed by CLI, TUI, run log, evals
  runlog.py        JSONL audit log
  evals.py         reliability harness
  git_ops.py       branch / commit / diff — used by the CLI layer only; the graph stays VCS-agnostic
  config.py        pydantic-settings
  prompts.py       system prompt + OpenAI-style tool schemas
  cli.py           run / chat / tui / eval
  tui.py           Textual UI — pure presentation layer
  llm/
    base.py          LLMClient protocol + ChatResponse (with token usage)
    fake_client.py    scripted responses, no network — what makes graph tests free
    groq_client.py    default provider
    gemini_client.py  fallback provider
demo_repo*/        seeded bugs: easy / logic / multifile / git (own git history; gitignored from this repo on purpose)
tests/             offline test suite, no API key needed
```

The graph has no idea what LLM it's talking to (`LLMClient` is a one-method protocol), which is what lets the tests drive the entire control flow — success, failure at `max_iterations`, premature `done`, test-tampering, truncation, compaction — with a `FakeLLMClient`.

## Development

```bash
uv run pytest                      # offline suite
uv run ruff check . && uv run ruff format --check . && uv run mypy
RUN_DOCKER_TESTS=1 uv run pytest tests/test_sandbox.py    # real container isolation checks (needs Docker)
uv run pytest --cov=agent          # coverage
```

## Things that broke, and what that taught me

Building the graph and passing the fake-client tests was the easy 80%. Pointing it at a live model, and then trying to make it safe, surfaced problems no mocking would have caught:

- **My first "sandbox" wasn't one.** I'd allowlisted `pytest` and `python` and called it sandboxed — but both run arbitrary code, so the allowlist stopped `rm` and nothing else. Real isolation needed a container: no network, read-only root, non-root, capabilities dropped. I now test those properties against a real container instead of trusting the flags.
- **The easiest way to "make tests pass" is to edit the tests.** Nothing stopped that. Now test files are read-only, and a hash snapshot catches tampering done through `python -c` too, refusing `done`.
- **Context grows until the provider says no.** Groq's free tier caps tokens per minute; whole-file reads and rewrites blew through it. Truncation, history compaction, line-range reads and snippet edits keep each request small — runs average ~14k tokens total.
- **Models get retired.** Both `llama-3.3-70b-versatile` (Groq) and `gemini-2.0-flash` (Gemini) were gone by the time I ran this for real.
- **A homemade message field isn't a wire format.** I'd put tool-call info under a made-up `tool_call` key; Groq's OpenAI-compatible endpoint rejected it. The fix was a real `tool_calls` array plus a matching `tool_call_id` on the result.
- **Free tiers have sharp edges.** Gemini's free tier caps at 20 requests/day — not enough for a multi-step loop — which is why Groq is the default.
- **Open models hallucinate tool names** (and invent arguments like `line_end`). Recovering gracefully — feeding the error back as an observation — matters more than preventing it.
- **Logging and TUIs don't mix.** Any stray write to stderr corrupts a full-screen terminal app, so console logging is disabled for `tui`; and enabling `logging.basicConfig()` also turned on `httpx`'s request logs until I capped third-party loggers.

## Limitations

- Python + pytest only, one repo per run. It creates branches and commits but never pushes or opens PRs.
- The Docker sandbox has `pytest` only and no network: a target repo with third-party dependencies needs `--sandbox local` (and a repo you trust).
- Test protection stops tampering with tests, not an implementation that special-cases them.
- The Gemini adapter is covered by stubbed-SDK tests but has had far less live use than Groq (its free tier is tiny).
- Success rates above come from a handful of small bugs on one model — evidence the loop works, not a benchmark.
