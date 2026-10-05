from pathlib import Path

import pytest

from agent import integrity


@pytest.mark.parametrize(
    "path",
    [
        "tests/test_a.py",
        "tests/helpers/data.json",
        "pkg/tests/anything.txt",
        "test/foo.py",
        "test_calc.py",
        "calc_test.py",
        "conftest.py",
        "pkg/conftest.py",
    ],
)
def test_is_test_path_true(path: str):
    assert integrity.is_test_path(path)


@pytest.mark.parametrize(
    "path",
    ["calculator.py", "src/app.py", "README.md", "tests/__pycache__/x.pyc", ".pytest_cache/v/cache", "latest.py"],
)
def test_is_test_path_false(path: str):
    assert not integrity.is_test_path(path)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "app.py").write_text("x = 1\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text("def test_x():\n    assert True\n")
    return tmp_path


def test_is_protected_write_resolves_dotdot_and_dot(repo: Path):
    assert integrity.is_protected_write(repo, "tests/test_app.py")
    assert integrity.is_protected_write(repo, "./tests/../tests/test_app.py")
    assert integrity.is_protected_write(repo, "tests/brand_new.py")
    assert not integrity.is_protected_write(repo, "app.py")


def test_is_protected_write_false_when_path_escapes_repo(repo: Path):
    assert not integrity.is_protected_write(repo, "../outside/tests/test_x.py")


def test_snapshot_ignores_caches_and_non_tests(repo: Path):
    (repo / "tests" / "__pycache__").mkdir()
    (repo / "tests" / "__pycache__" / "test_app.cpython-311.pyc").write_bytes(b"\x00")
    assert set(integrity.snapshot_tests(repo)) == {"tests/test_app.py"}


def test_changed_tests_reports_modified_added_deleted(repo: Path):
    snapshot = integrity.snapshot_tests(repo)
    assert integrity.changed_tests(repo, snapshot) == []

    (repo / "tests" / "test_app.py").write_text("# changed\n")
    (repo / "tests" / "test_new.py").write_text("# new\n")
    assert integrity.changed_tests(repo, snapshot) == [
        "tests/test_app.py (modified)",
        "tests/test_new.py (added)",
    ]

    (repo / "tests" / "test_app.py").unlink()
    assert "tests/test_app.py (deleted)" in integrity.changed_tests(repo, snapshot)
