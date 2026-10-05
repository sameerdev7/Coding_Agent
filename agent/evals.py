"""Reliability harness: run the agent N times per repo and report real numbers, not anecdotes."""

import json
import shutil
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from agent import tools
from agent.events import AgentRun
from agent.graph import build_graph, initial_state, is_success
from agent.llm.base import LLMClient
from agent.sandbox import Sandbox

DEFAULT_TASK = "Make all tests in tests/ pass"


@dataclass
class EvalResult:
    repo: str
    run: int
    success: bool
    iterations: int
    total_tokens: int
    seconds: float
    error: str | None = None


def _copy_repo(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))


def _run_one(
    repo: Path,
    run_number: int,
    llm_factory: Callable[[], LLMClient],
    sandbox: Sandbox | None,
    max_iterations: int,
    protect_tests: bool,
    task: str,
) -> EvalResult:
    started = time.monotonic()

    def elapsed() -> float:
        return round(time.monotonic() - started, 2)

    with tempfile.TemporaryDirectory(prefix="agent-eval-") as tmp:
        workdir = Path(tmp) / repo.name
        _copy_repo(repo, workdir)

        baseline = tools.run_command(workdir, ["pytest", "-q"], sandbox=sandbox)
        if baseline.startswith("exit=0"):
            return EvalResult(
                repo.name, run_number, False, 0, 0, elapsed(), "baseline already passes — reset the demo repo"
            )

        try:
            graph = build_graph(llm_factory(), sandbox=sandbox, protect_tests=protect_tests)
            run = AgentRun(graph, initial_state(task, workdir, max_iterations))
            for _ in run.events():
                pass
            final = run.final_state
            # Verify independently: the agent's own "done" is a claim, not evidence.
            verified = tools.run_command(workdir, ["pytest", "-q"], sandbox=sandbox).startswith("exit=0")
            return EvalResult(
                repo.name,
                run_number,
                is_success(final) and verified,
                final["iterations"],
                final["total_tokens"],
                elapsed(),
            )
        except Exception as exc:
            return EvalResult(repo.name, run_number, False, 0, 0, elapsed(), f"{type(exc).__name__}: {exc}")


def run_eval(
    repos: Sequence[Path],
    runs: int,
    llm_factory: Callable[[], LLMClient],
    sandbox: Sandbox | None = None,
    max_iterations: int = 15,
    protect_tests: bool = True,
    task: str = DEFAULT_TASK,
    on_result: Callable[[EvalResult], None] | None = None,
) -> list[EvalResult]:
    results: list[EvalResult] = []
    for repo in repos:
        for run_number in range(1, runs + 1):
            result = _run_one(repo, run_number, llm_factory, sandbox, max_iterations, protect_tests, task)
            results.append(result)
            if on_result:
                on_result(result)
    return results


def _row(label: str, results: list[EvalResult]) -> dict[str, Any]:
    successes = [r for r in results if r.success]
    return {
        "repo": label,
        "runs": len(results),
        "successes": len(successes),
        "success_rate": len(successes) / len(results) if results else 0.0,
        "avg_iterations_on_success": round(float(mean(r.iterations for r in successes)), 1) if successes else None,
        "avg_tokens": round(float(mean(r.total_tokens for r in results))) if results else 0,
        "avg_seconds": round(mean(r.seconds for r in results), 1) if results else 0.0,
        "errors": sum(1 for r in results if r.error),
    }


def summarize(results: Sequence[EvalResult]) -> list[dict[str, Any]]:
    by_repo: dict[str, list[EvalResult]] = {}
    for result in results:
        by_repo.setdefault(result.repo, []).append(result)
    rows = [_row(repo, repo_results) for repo, repo_results in by_repo.items()]
    if len(rows) > 1:
        rows.append(_row("**overall**", list(results)))
    return rows


def to_markdown(rows: Sequence[dict[str, Any]]) -> str:
    header = "| repo | runs | success | avg iterations (on success) | avg tokens | avg seconds | errors |"
    divider = "|---|---|---|---|---|---|---|"
    lines = [header, divider]
    for row in rows:
        iterations = "-" if row["avg_iterations_on_success"] is None else row["avg_iterations_on_success"]
        lines.append(
            f"| {row['repo']} | {row['runs']} | {row['successes']}/{row['runs']} ({row['success_rate']:.0%}) "
            f"| {iterations} | {row['avg_tokens']:,} | {row['avg_seconds']} | {row['errors']} |"
        )
    return "\n".join(lines)


def save_results(path: Path, results: Sequence[EvalResult], meta: dict[str, Any]) -> None:
    payload = {"meta": meta, "results": [asdict(r) for r in results], "summary": summarize(results)}
    path.write_text(json.dumps(payload, indent=2))
