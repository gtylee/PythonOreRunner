from __future__ import annotations

from pathlib import Path
import sys


def _bootstrap() -> None:
    here = Path(__file__).resolve()
    py_root = here.parents[2]
    if str(py_root) not in sys.path:
        sys.path.insert(0, str(py_root))


def main() -> int:
    _bootstrap()
    from native_xva_interface.large_fx_universe_benchmark import main as benchmark_main

    return benchmark_main()


if __name__ == "__main__":
    raise SystemExit(main())
