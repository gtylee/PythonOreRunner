"""Compatibility wrapper for canonical HW2F ORE runner."""

from pythonore.hw2f_ore_runner import *  # noqa: F401,F403
from pythonore.hw2f_ore_runner import main


if __name__ == "__main__":
    raise SystemExit(main())
