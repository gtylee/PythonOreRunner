from __future__ import annotations

from pathlib import Path

from py_ore_tools import repo_paths


def test_find_engine_repo_root_honors_env(monkeypatch, tmp_path: Path):
    (tmp_path / "Examples").mkdir()
    monkeypatch.setenv("ENGINE_REPO_ROOT", str(tmp_path))
    assert repo_paths.find_engine_repo_root() == tmp_path.resolve()


def test_find_examples_repo_root_prefers_local_examples():
    assert repo_paths.find_examples_repo_root() == repo_paths.pythonorerunner_root()
