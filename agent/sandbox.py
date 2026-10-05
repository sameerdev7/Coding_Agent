"""Where `run_command` actually executes.

`LocalSandbox` runs on the host (allowlist + timeout only — no isolation: `python` and `pytest`
both execute arbitrary code). `DockerSandbox` runs the same command in a locked-down container.
"""

import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

SANDBOX_IMAGE = "coding-agent-sandbox:1"
SANDBOX_DOCKERFILE = "FROM python:3.11-slim\nRUN pip install --no-cache-dir pytest\nWORKDIR /workspace\n"


def format_result(returncode: int, stdout: str, stderr: str) -> str:
    return f"exit={returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"


class Sandbox(Protocol):
    name: str

    def run(self, repo_root: Path, args: list[str], timeout: int) -> str: ...


class LocalSandbox:
    name = "local (NO isolation: allowlist and timeout only)"

    def run(self, repo_root: Path, args: list[str], timeout: int) -> str:
        try:
            result = subprocess.run(
                args,
                cwd=repo_root,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
                shell=False,
            )
        except subprocess.TimeoutExpired:
            return f"exit=timeout after {timeout}s"
        return format_result(result.returncode, result.stdout, result.stderr)


class DockerSandbox:
    name = "docker (no network, read-only rootfs, non-root, no capabilities, resource-limited)"

    def __init__(self, image: str = SANDBOX_IMAGE, memory: str = "512m", cpus: str = "1", pids_limit: int = 128):
        self.image = image
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit

    def build_command(self, repo_root: Path, args: list[str], container_name: str) -> list[str]:
        user = f"{os.getuid()}:{os.getgid()}" if hasattr(os, "getuid") else "1000:1000"
        return [
            "docker", "run", "--rm",
            "--name", container_name,
            "--network", "none",
            "--memory", self.memory,
            "--memory-swap", self.memory,
            "--cpus", self.cpus,
            "--pids-limit", str(self.pids_limit),
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--read-only",
            "--tmpfs", "/tmp",
            "--user", user,
            "-e", "HOME=/tmp",
            "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-v", f"{repo_root.resolve()}:/workspace",
            "-w", "/workspace",
            self.image,
            *args,
        ]  # fmt: skip

    def ensure_image(self) -> None:
        inspect = subprocess.run(["docker", "image", "inspect", self.image], capture_output=True)
        if inspect.returncode == 0:
            return
        logger.info("building sandbox image %s (one-time)...", self.image)
        build = subprocess.run(
            ["docker", "build", "-t", self.image, "-"],
            input=SANDBOX_DOCKERFILE,
            text=True,
            capture_output=True,
        )
        if build.returncode != 0:
            raise RuntimeError(f"failed to build sandbox image: {build.stderr.strip()}")

    def run(self, repo_root: Path, args: list[str], timeout: int) -> str:
        container_name = f"coding-agent-{uuid.uuid4().hex[:12]}"
        command = self.build_command(repo_root, args, container_name)
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
                shell=False,
            )
        except subprocess.TimeoutExpired:
            # Killing the docker client does not stop the container.
            subprocess.run(["docker", "kill", container_name], capture_output=True)
            return f"exit=timeout after {timeout}s"
        return format_result(result.returncode, result.stdout, result.stderr)


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def make_sandbox(mode: str) -> Sandbox:
    """mode: 'docker' (fail if unavailable), 'local', or 'auto' (docker if reachable, else local)."""
    if mode == "local":
        return LocalSandbox()
    available = docker_available()
    if mode == "docker" and not available:
        raise RuntimeError("sandbox 'docker' was requested but the docker daemon is not reachable")
    if available:
        sandbox = DockerSandbox()
        sandbox.ensure_image()
        return sandbox
    return LocalSandbox()
