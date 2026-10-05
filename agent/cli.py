import argparse
import json
import logging
import sys
import time
from pathlib import Path

from agent import git_ops
from agent.config import Settings
from agent.evals import DEFAULT_TASK, EvalResult, run_eval, save_results, summarize, to_markdown
from agent.events import AgentRun, PlanEvent, StepEvent
from agent.graph import build_graph, continue_state, initial_state, is_success
from agent.llm.base import LLMClient
from agent.runlog import RunLogger
from agent.sandbox import Sandbox, make_sandbox
from agent.state import WRITE_TOOLS, AgentState

DEFAULT_EVAL_REPOS = ["demo_repo", "demo_repo_logic", "demo_repo_multifile", "demo_repo_git"]


def _configure_logging(enable_console: bool) -> None:
    """The TUI owns the whole screen — any stray stdout/stderr write corrupts its rendering,
    so console logging is only safe for non-TUI commands."""
    if enable_console:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        for noisy in ("httpx", "httpcore", "groq", "google_genai"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    else:
        logging.getLogger().addHandler(logging.NullHandler())


def build_llm(settings: Settings, provider: str) -> LLMClient:
    api_key = settings.require_key_for(provider)
    if provider == "groq":
        from agent.llm.groq_client import GroqClient

        return GroqClient(api_key=api_key)
    if provider == "gemini":
        from agent.llm.gemini_client import GeminiClient

        return GeminiClient(api_key=api_key)
    raise ValueError(f"unknown provider: {provider}")


def build_sandbox(mode: str) -> Sandbox:
    sandbox = make_sandbox(mode)
    icon = "🔒" if sandbox.name.startswith("docker") else "⚠️ "
    print(f"{icon} sandbox: {sandbox.name}")
    return sandbox


def _start_git_branch(repo_root: Path, task: str, use_git: bool) -> str | None:
    """Returns the branch name if git integration is active for this run, else None."""
    if not use_git or not git_ops.is_git_repo(repo_root):
        return None
    if not git_ops.is_clean(repo_root):
        print("⚠️  repo has uncommitted changes — skipping git branch/commit for this run.")
        return None
    branch = git_ops.create_agent_branch(repo_root, task)
    print(f"🌿 working on branch {branch}")
    return branch


def _execute(graph, state: AgentState, repo_root: Path, git_active: bool, logger: RunLogger) -> AgentState:
    run = AgentRun(graph, state)
    for event in run.events():
        if isinstance(event, PlanEvent):
            print(f"📋 plan:\n{event.plan}\n")
            logger.log("plan", plan=event.plan)
        elif isinstance(event, StepEvent):
            if event.tool:
                print(f"[{event.step}] plan     -> {event.tool}({event.args})")
            if event.result:
                first_line = event.result.splitlines()[0] if event.result.splitlines() else event.result
                print(f"[{event.step}] observe  -> {first_line}")
            if git_active and event.tool in WRITE_TOOLS:
                diff = git_ops.diff_for_path(repo_root, str(event.args.get("path", "")))
                if diff:
                    print(diff)
            logger.log(
                "step",
                step=event.step,
                tool=event.tool,
                args=event.args,
                result=event.result,
                tests_passing=event.tests_passing,
                total_tokens=event.total_tokens,
            )
    return run.final_state


def _report_outcome(final_state: AgentState, repo_root: Path, branch: str | None, logger: RunLogger, started: float):
    success = is_success(final_state)

    if branch:
        if success:
            message = final_state.get("summary") or "automated fix"
            git_ops.commit_all(repo_root, f"agent: {message}")
            print(f"✅ committed to {branch}")
        else:
            print(f"⚠️  left uncommitted on {branch} for inspection — fix didn't verify, nothing was committed.")

    logger.log(
        "run_end",
        success=success,
        iterations=final_state["iterations"],
        total_tokens=final_state["total_tokens"],
        seconds=round(time.monotonic() - started, 2),
        branch=branch,
    )

    tokens = f"{final_state['total_tokens']:,} tokens"
    if success:
        print(f"\n✅ Task complete in {final_state['iterations']} iterations ({tokens}). Tests passing.")
        return 0
    print(
        f"\n❌ Task failed after {final_state['iterations']} iterations ({tokens}). "
        f"Tests passing: {final_state['tests_passing']}"
    )
    return 1


def _resolve_repo(repo: str) -> Path | None:
    repo_root = Path(repo).resolve()
    if not repo_root.is_dir():
        print(f"error: repo path does not exist: {repo_root}", file=sys.stderr)
        return None
    return repo_root


def run(args: argparse.Namespace, settings: Settings) -> int:
    repo_root = _resolve_repo(args.repo)
    if repo_root is None:
        return 1

    sandbox = build_sandbox(args.sandbox)
    branch = _start_git_branch(repo_root, args.task, not args.no_git)
    graph = build_graph(build_llm(settings, args.provider), sandbox=sandbox, protect_tests=not args.allow_test_edits)
    state = initial_state(args.task, repo_root, max_iterations=args.max_iterations)

    started = time.monotonic()
    with RunLogger(Path(args.run_log) if args.run_log else None) as logger:
        logger.log("run_start", task=args.task, repo=str(repo_root), provider=args.provider, sandbox=sandbox.name)
        final_state = _execute(graph, state, repo_root, branch is not None, logger)
        if args.transcript_out:
            Path(args.transcript_out).write_text(json.dumps(final_state["messages"], indent=2, default=str))
        return _report_outcome(final_state, repo_root, branch, logger, started)


def chat(args: argparse.Namespace, settings: Settings) -> int:
    repo_root = _resolve_repo(args.repo)
    if repo_root is None:
        return 1

    sandbox = build_sandbox(args.sandbox)
    graph = build_graph(build_llm(settings, args.provider), sandbox=sandbox, protect_tests=not args.allow_test_edits)

    print(f"Interactive mode — repo: {repo_root}. Type a task, or 'exit' to quit.")
    branch: str | None = None
    state: AgentState | None = None
    with RunLogger(Path(args.run_log) if args.run_log else None) as logger:
        while True:
            try:
                task = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if task.lower() in {"exit", "quit"}:
                break
            if not task:
                continue

            if branch is None:
                branch = _start_git_branch(repo_root, task, not args.no_git)

            state = (
                initial_state(task, repo_root, args.max_iterations) if state is None else continue_state(state, task)
            )
            logger.log("run_start", task=task, repo=str(repo_root), provider=args.provider, sandbox=sandbox.name)
            started = time.monotonic()
            state = _execute(graph, state, repo_root, branch is not None, logger)
            _report_outcome(state, repo_root, branch, logger, started)

    print("Goodbye.")
    return 0


def tui(args: argparse.Namespace, settings: Settings) -> int:
    repo_root = _resolve_repo(args.repo)
    if repo_root is None:
        return 1

    from agent.tui import CodingAgentTUI

    sandbox = build_sandbox(args.sandbox)
    llm = build_llm(settings, args.provider)
    with RunLogger(Path(args.run_log) if args.run_log else None) as logger:
        CodingAgentTUI(
            llm,
            repo_root,
            args.max_iterations,
            not args.no_git,
            sandbox=sandbox,
            protect_tests=not args.allow_test_edits,
            run_logger=logger,
        ).run()
    return 0


def evaluate(args: argparse.Namespace, settings: Settings) -> int:
    repos = [Path(r).resolve() for r in (args.repos or [r for r in DEFAULT_EVAL_REPOS if Path(r).is_dir()])]
    missing = [str(r) for r in repos if not r.is_dir()]
    if missing:
        print(f"error: repo path(s) do not exist: {', '.join(missing)}", file=sys.stderr)
        return 1

    sandbox = build_sandbox(args.sandbox)

    def show(result: EvalResult) -> None:
        status = "✅" if result.success else "❌"
        note = f"  ({result.error})" if result.error else ""
        print(
            f"  {status} {result.repo} #{result.run}: {result.iterations} iterations, "
            f"{result.total_tokens:,} tokens, {result.seconds}s{note}"
        )

    print(f"evaluating {len(repos)} repo(s) x {args.runs} run(s) on {args.provider}\n")
    results = run_eval(
        repos,
        args.runs,
        lambda: build_llm(settings, args.provider),
        sandbox=sandbox,
        max_iterations=args.max_iterations,
        protect_tests=not args.allow_test_edits,
        task=args.task,
        on_result=show,
    )

    rows = summarize(results)
    print("\n" + to_markdown(rows))
    if args.out:
        meta = {"provider": args.provider, "sandbox": sandbox.name, "runs_per_repo": args.runs, "task": args.task}
        save_results(Path(args.out), results, meta)
        print(f"\nsaved {args.out}")

    overall = rows[-1]["success_rate"] if rows else 0.0
    return 0 if overall >= args.min_success_rate else 1


def _add_common_args(parser: argparse.ArgumentParser, settings: Settings) -> None:
    parser.add_argument("--max-iterations", type=int, default=settings.max_iterations)
    parser.add_argument("--provider", choices=["groq", "gemini"], default=settings.llm_provider)
    parser.add_argument(
        "--sandbox",
        choices=["auto", "docker", "local"],
        default=settings.sandbox,
        help="Where run_command executes: docker (isolated), local (NO isolation), auto (docker if available).",
    )
    parser.add_argument(
        "--allow-test-edits",
        action="store_true",
        default=not settings.protect_tests,
        help="Let the agent modify test files (they are read-only by default).",
    )


def _add_session_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", required=True, help="Path to the target repository.")
    parser.add_argument("--no-git", action="store_true", help="Never create a branch or commit, even in a git repo.")
    parser.add_argument("--run-log", default=None, help="Append a JSONL audit log of this session to this path.")


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="coding-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the agent against a repository, once.")
    _add_session_args(run_parser)
    _add_common_args(run_parser, settings)
    run_parser.add_argument("--task", required=True, help="Natural-language task description.")
    run_parser.add_argument("--transcript-out", default=None, help="Path to dump the full message history as JSON.")

    chat_parser = subparsers.add_parser("chat", help="Interactive mode: give follow-up tasks in the same session.")
    _add_session_args(chat_parser)
    _add_common_args(chat_parser, settings)

    tui_parser = subparsers.add_parser("tui", help="Interactive terminal UI, with live panels instead of plain prints.")
    _add_session_args(tui_parser)
    _add_common_args(tui_parser, settings)

    eval_parser = subparsers.add_parser("eval", help="Run the agent N times per repo and report success rate.")
    _add_common_args(eval_parser, settings)
    eval_parser.add_argument("--repos", nargs="*", help="Repos to evaluate (default: the bundled demo repos).")
    eval_parser.add_argument("--runs", type=int, default=3, help="Runs per repo.")
    eval_parser.add_argument("--task", default=DEFAULT_TASK)
    eval_parser.add_argument("--out", default=None, help="Write full results + summary as JSON to this path.")
    eval_parser.add_argument("--min-success-rate", type=float, default=0.0, help="Exit non-zero below this rate.")

    return parser


def main(argv: list[str] | None = None) -> int:
    settings = Settings()
    args = build_parser(settings).parse_args(argv)
    _configure_logging(enable_console=args.command != "tui")

    handlers = {"run": run, "chat": chat, "tui": tui, "eval": evaluate}
    try:
        return handlers[args.command](args, settings)
    except (ValueError, RuntimeError) as exc:  # config / environment problems, not bugs
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
