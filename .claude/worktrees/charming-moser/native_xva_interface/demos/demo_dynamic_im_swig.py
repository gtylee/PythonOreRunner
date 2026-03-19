from __future__ import annotations

from pathlib import Path
import sys


def _bootstrap() -> Path:
    here = Path(__file__).resolve()
    repo_root = here.parents[4]
    py_root = repo_root / "Tools" / "PythonOreRunner"
    if str(py_root) not in sys.path:
        sys.path.insert(0, str(py_root))
    return repo_root


def main() -> None:
    repo_root = _bootstrap()

    from native_xva_interface import ORESwigAdapter, XVAEngine, XVALoader

    input_dir = repo_root / "Examples" / "InitialMargin" / "Input" / "Dim2"
    snapshot = XVALoader.from_files(str(input_dir), ore_file="ore_dynamicsimm.xml")

    result = XVAEngine(adapter=ORESwigAdapter()).create_session(snapshot).run(return_cubes=False)
    print("dim_mode:", result.metadata.get("dim_mode"))
    print("dim_report_source:", result.metadata.get("dim_report_source"))
    print("xva_by_metric:", result.xva_by_metric)
    print("dim_reports:", [name for name in ("dim_evolution", "dim_regression") if name in result.reports])


if __name__ == "__main__":
    main()
