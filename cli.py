import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from agent.graph import build_graph, initial_state


def build_llm(provider: str):
    if provider == "groq":
        from agent.llm.groq_client import GroqClient

        return GroqClient()
    if provider == "gemini":
        from agent.llm.gemini_client import GeminiClient

        return GeminiClient()
    raise ValueError(f"unknown provider: {provider}")


def run(repo: str, task: str, max_iterations: int, provider: str, transcript_out: str | None) -> int:
    load_dotenv()
    repo_root = Path(repo).resolve()
    if not repo_root.is_dir():
        print(f"error: repo path does not exist: {repo_root}", file=sys.stderr)
        return 1

    llm = build_llm(provider)
    graph = build_graph(llm)
    state = initial_state(task, repo_root, max_iterations=max_iterations)

    step = 0
    final_state = state
    for event in graph.stream(state, stream_mode="values"):
        final_state = event
        iterations = event["iterations"]
        if iterations != step:
            step = iterations
            call = event.get("pending_tool_call")
            if call:
                print(f"[{step}] plan     -> {call['name']}({call['args']})")
            result = event.get("last_tool_result")
            if result:
                first_line = result.splitlines()[0] if result.splitlines() else result
                print(f"[{step}] observe  -> {first_line}")

    if transcript_out:
        Path(transcript_out).write_text(json.dumps(final_state["messages"], indent=2, default=str))

    if final_state["finished"] and final_state["tests_passing"]:
        print(f"\n✅ Task complete in {final_state['iterations']} iterations. Tests passing.")
        return 0

    print(f"\n❌ Task failed after {final_state['iterations']} iterations. Tests passing: {final_state['tests_passing']}")
    return 1


def main() -> int:
    load_dotenv()
    default_provider = os.environ.get("LLM_PROVIDER", "groq")

    parser = argparse.ArgumentParser(prog="python -m cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the agent against a repository.")
    run_parser.add_argument("--repo", required=True, help="Path to the target repository.")
    run_parser.add_argument("--task", required=True, help="Natural-language task description.")
    run_parser.add_argument("--max-iterations", type=int, default=15)
    run_parser.add_argument("--provider", choices=["groq", "gemini"], default=default_provider)
    run_parser.add_argument("--transcript-out", default=None, help="Path to dump the full message history as JSON.")

    args = parser.parse_args()
    if args.command == "run":
        return run(args.repo, args.task, args.max_iterations, args.provider, args.transcript_out)
    return 1


if __name__ == "__main__":
    sys.exit(main())
