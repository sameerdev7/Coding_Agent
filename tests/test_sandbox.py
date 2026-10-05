import os
import subprocess
from pathlib import Path

import pytest

from agent import sandbox as sandbox_module
from agent.sandbox import DockerSandbox, LocalSandbox, make_sandbox


def test_local_sandbox_runs_a_command(tmp_path: Path):
    result = LocalSandbox().run(tmp_path, ["python3", "-c", "print('hi')"], timeout=10)
    assert result.startswith("exit=0")
    assert "hi" in result


def test_local_sandbox_timeout(tmp_path: Path):
    result = LocalSandbox().run(tmp_path, ["python3", "-c", "import time; time.sleep(5)"], timeout=1)
    assert result == "exit=timeout after 1s"


def test_docker_command_is_locked_down(tmp_path: Path):
    command = DockerSandbox().build_command(tmp_path, ["pytest", "-q"], "coding-agent-test")
    joined = " ".join(command)

    assert command[:3] == ["docker", "run", "--rm"]
    assert "--network none" in joined
    assert "--cap-drop ALL" in joined
    assert "--security-opt no-new-privileges" in joined
    assert "--read-only" in command
    assert "--memory 512m" in joined and "--pids-limit 128" in joined
    assert f"-v {tmp_path.resolve()}:/workspace" in joined
    assert command[-3:] == [sandbox_module.SANDBOX_IMAGE, "pytest", "-q"][-3:] or command[-2:] == ["pytest", "-q"]
    assert "--privileged" not in command


def test_docker_command_runs_as_the_host_user_not_root(tmp_path: Path):
    command = DockerSandbox().build_command(tmp_path, ["pytest"], "n")
    user = command[command.index("--user") + 1]
    assert user != "0:0" and not user.startswith("0:")


def test_docker_timeout_kills_the_container(tmp_path: Path, monkeypatch):
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["docker", "run"]:
            raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(sandbox_module.subprocess, "run", fake_run)

    result = DockerSandbox().run(tmp_path, ["pytest"], timeout=3)

    assert result == "exit=timeout after 3s"
    run_cmd = next(c for c in calls if c[:2] == ["docker", "run"])
    kill_cmd = next(c for c in calls if c[:2] == ["docker", "kill"])
    assert kill_cmd[2] == run_cmd[run_cmd.index("--name") + 1]


def test_make_sandbox_local_never_touches_docker(monkeypatch):
    monkeypatch.setattr(sandbox_module, "docker_available", lambda: pytest.fail("should not probe docker"))
    assert isinstance(make_sandbox("local"), LocalSandbox)


def test_make_sandbox_docker_fails_loudly_when_unavailable(monkeypatch):
    monkeypatch.setattr(sandbox_module, "docker_available", lambda: False)
    with pytest.raises(RuntimeError, match="not reachable"):
        make_sandbox("docker")


def test_make_sandbox_auto_falls_back_to_local(monkeypatch):
    monkeypatch.setattr(sandbox_module, "docker_available", lambda: False)
    assert isinstance(make_sandbox("auto"), LocalSandbox)


@pytest.fixture(scope="module")
def docker_sandbox() -> DockerSandbox:
    sandbox = DockerSandbox()
    sandbox.ensure_image()
    return sandbox


@pytest.mark.skipif(os.environ.get("RUN_DOCKER_TESTS") != "1", reason="set RUN_DOCKER_TESTS=1 to run (needs docker)")
class TestDockerIntegration:
    def test_runs_pytest_inside_the_container(self, docker_sandbox: DockerSandbox, tmp_path: Path):
        (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert True\n")
        result = docker_sandbox.run(tmp_path, ["pytest", "-q"], timeout=60)
        assert result.startswith("exit=0"), result

    def test_has_no_network(self, docker_sandbox: DockerSandbox, tmp_path: Path):
        code = "import socket; socket.create_connection(('1.1.1.1', 80), timeout=3)"
        result = docker_sandbox.run(tmp_path, ["python", "-c", code], timeout=30)
        assert not result.startswith("exit=0")

    def test_cannot_write_outside_the_workspace(self, docker_sandbox: DockerSandbox, tmp_path: Path):
        code = "open('/etc/pwned', 'w').write('x')"
        result = docker_sandbox.run(tmp_path, ["python", "-c", code], timeout=30)
        assert not result.startswith("exit=0")

    def test_cannot_see_host_files_outside_the_mount(self, docker_sandbox: DockerSandbox, tmp_path: Path):
        secret = tmp_path.parent / "host_secret.txt"
        secret.write_text("secret")
        code = f"print(open({str(secret)!r}).read())"
        result = docker_sandbox.run(tmp_path, ["python", "-c", code], timeout=30)
        assert not result.startswith("exit=0")

    def test_timeout_is_enforced(self, docker_sandbox: DockerSandbox, tmp_path: Path):
        result = docker_sandbox.run(tmp_path, ["python", "-c", "import time; time.sleep(30)"], timeout=2)
        assert result == "exit=timeout after 2s"
