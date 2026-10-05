import time
from pathlib import Path
from typing import ClassVar

from textual import work
from textual.app import App, ComposeResult
from textual.binding import BindingType
from textual.widgets import Footer, Header, Input, RichLog

from agent import git_ops
from agent.events import AgentRun, PlanEvent, StepEvent
from agent.graph import build_graph, continue_state, initial_state, is_success
from agent.llm.base import LLMClient
from agent.runlog import RunLogger
from agent.sandbox import Sandbox
from agent.state import WRITE_TOOLS, AgentState


class CodingAgentTUI(App):
    """Interactive terminal UI over the same graph/tools the CLI uses — pure presentation layer."""

    CSS = """
    RichLog { border: round $accent; padding: 0 1; }
    Input { dock: bottom; margin: 0 1 1 1; }
    """
    BINDINGS: ClassVar[list[BindingType]] = [("q", "quit", "Quit")]

    def __init__(
        self,
        llm: LLMClient,
        repo_root: Path,
        max_iterations: int,
        use_git: bool,
        sandbox: Sandbox | None = None,
        protect_tests: bool = True,
        run_logger: RunLogger | None = None,
    ):
        super().__init__()
        self.repo_root = repo_root
        self.max_iterations = max_iterations
        self.use_git = use_git
        self.sandbox = sandbox
        self.run_logger = run_logger or RunLogger(None)
        self.graph = build_graph(llm, sandbox=sandbox, protect_tests=protect_tests)
        self.state: AgentState | None = None
        self.branch: str | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield RichLog(id="log", markup=True, wrap=True, auto_scroll=True)
        yield Input(placeholder="Type a task and press Enter...", id="task_input")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Coding Agent"
        self._update_status(0, None, 0)
        log = self.query_one("#log", RichLog)
        log.write(f"[bold]repo:[/bold] {self.repo_root}")
        if self.sandbox is not None:
            colour = "green" if self.sandbox.name.startswith("docker") else "yellow"
            log.write(f"[{colour}]sandbox: {self.sandbox.name}[/{colour}]")
        log.write("Type a task below and press Enter. Follow-up tasks continue the same conversation.\n")
        self.query_one("#task_input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        task = event.value.strip()
        if not task:
            return
        event.input.value = ""
        event.input.disabled = True
        self.run_task(task)

    @work(thread=True)
    def run_task(self, task: str) -> None:
        log = self.query_one("#log", RichLog)
        write = lambda text: self.call_from_thread(log.write, text)  # noqa: E731
        started = time.monotonic()
        write(f"\n[bold magenta]> {task}[/bold magenta]")

        if self.branch is None and self.use_git and git_ops.is_git_repo(self.repo_root):
            if git_ops.is_clean(self.repo_root):
                self.branch = git_ops.create_agent_branch(self.repo_root, task)
                write(f"[green]🌿 branch: {self.branch}[/green]")
            else:
                write("[yellow]⚠️  repo has uncommitted changes — skipping git workflow[/yellow]")

        self.state = (
            initial_state(task, self.repo_root, self.max_iterations)
            if self.state is None
            else continue_state(self.state, task)
        )
        self.run_logger.log("run_start", task=task, repo=str(self.repo_root))

        run = AgentRun(self.graph, self.state)
        try:
            for event in run.events():
                if isinstance(event, PlanEvent):
                    write(f"[bold cyan]📋 plan:[/bold cyan]\n{event.plan}\n")
                    self.run_logger.log("plan", plan=event.plan)
                elif isinstance(event, StepEvent):
                    self._show_step(event, write)
        except Exception as exc:
            write(f"[bold red]error: {type(exc).__name__}: {exc}[/bold red]\n")
            self.run_logger.log("run_error", error=f"{type(exc).__name__}: {exc}")
            self.call_from_thread(self._reenable_input)
            return

        final_state = run.final_state
        self.state = final_state
        success = is_success(final_state)

        if self.branch:
            if success:
                message = final_state.get("summary") or "automated fix"
                git_ops.commit_all(self.repo_root, f"agent: {message}")
                write(f"[green]✅ committed to {self.branch}[/green]")
            else:
                write(f"[yellow]⚠️  left uncommitted on {self.branch} for inspection[/yellow]")

        self.run_logger.log(
            "run_end",
            success=success,
            iterations=final_state["iterations"],
            total_tokens=final_state["total_tokens"],
            seconds=round(time.monotonic() - started, 2),
            branch=self.branch,
        )

        summary = f"{final_state['iterations']} iterations, {final_state['total_tokens']:,} tokens"
        if success:
            write(f"[bold green]✅ task complete in {summary}[/bold green]\n")
        else:
            write(f"[bold red]❌ task failed after {summary}[/bold red]\n")

        self.call_from_thread(self._reenable_input)

    def _show_step(self, event: StepEvent, write) -> None:
        self.call_from_thread(self._update_status, event.step, event.tests_passing, event.total_tokens)
        if event.tool:
            write(f"[yellow][{event.step}] plan[/yellow]     -> {event.tool}({event.args})")
        if event.result:
            first_line = event.result.splitlines()[0] if event.result.splitlines() else event.result
            write(f"[blue][{event.step}] observe[/blue]  -> {first_line}")
        if self.branch and event.tool in WRITE_TOOLS:
            diff = git_ops.diff_for_path(self.repo_root, str(event.args.get("path", "")))
            if diff:
                write(diff)
        self.run_logger.log(
            "step",
            step=event.step,
            tool=event.tool,
            args=event.args,
            result=event.result,
            tests_passing=event.tests_passing,
            total_tokens=event.total_tokens,
        )

    def _update_status(self, step: int, tests_passing: bool | None, total_tokens: int) -> None:
        status = "✅" if tests_passing else ("❌" if tests_passing is False else "…")
        branch_note = f" | 🌿 {self.branch}" if self.branch else ""
        self.sub_title = (
            f"{self.repo_root.name}{branch_note} | iter {step}/{self.max_iterations} | "
            f"tests {status} | {total_tokens:,} tokens"
        )

    def _reenable_input(self) -> None:
        task_input = self.query_one("#task_input", Input)
        task_input.disabled = False
        task_input.focus()
