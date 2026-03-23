from __future__ import annotations

from pathlib import Path

from py_ore_tools import repo_paths


def test_find_engine_repo_root_honors_env(monkeypatch, tmp_path: Path):
    (tmp_path / "Examples").mkdir()
    monkeypatch.setenv("ENGINE_REPO_ROOT", str(tmp_path))
    assert repo_paths.find_engine_repo_root() == tmp_path.resolve()


def test_find_engine_repo_root_treats_invalid_env_as_authoritative(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ENGINE_REPO_ROOT", str(tmp_path / "missing"))
    assert repo_paths.find_engine_repo_root() is None


def test_find_examples_repo_root_prefers_local_examples():
    assert repo_paths.find_examples_repo_root() == repo_paths.pythonorerunner_root()


def test_find_ore_bin_honors_env(monkeypatch, tmp_path: Path):
    ore_bin = tmp_path / "ore"
    ore_bin.write_text("", encoding="utf-8")
    monkeypatch.setenv("ORE_EXE", str(ore_bin))
    assert repo_paths.find_ore_bin() == ore_bin.resolve()


def test_find_ore_bin_treats_invalid_env_as_authoritative(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ORE_EXE", str(tmp_path / "missing-ore"))
    assert repo_paths.find_ore_bin() is None
