import os
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent / "src"
if _SRC_ROOT.exists():
    src_root_text = str(_SRC_ROOT)
    if src_root_text not in sys.path:
        sys.path.insert(0, src_root_text)

from pythonore.repo_paths import find_engine_repo_root, find_ore_bin


def main() -> int:
    engine_root = find_engine_repo_root()
    ore_bin = find_ore_bin()
    if engine_root is None or ore_bin is None:
        print("example_basic.py requires a local ORE Engine checkout and built ore binary.")
        print("Set ENGINE_REPO_ROOT and/or ORE_EXE, or use the Python-only snapshot examples instead.")
        return 0

    from pythonore.ore import OreBasic

    example_dir = Path(os.environ.get("ORE_EXAMPLE_DIR", str(engine_root / "Examples" / "Legacy" / "Example_1")))
    my_ore = OreBasic.from_folders(
        input_folder=str(example_dir / "Input"),
        output_folder=str(example_dir / "Output"),
        execution_folder=str(example_dir),
    )
    my_ore.run(ore_exe=str(ore_bin), delete_output_folder_before_run=False)
    my_ore.parse_output()

    print("Run produced output files: " + str(len(my_ore.output.locations)))
    print(my_ore.output.csv["npv"][["#TradeId", "TradeType", "NPV(Base)"]])
    print(my_ore.output.csv["xva"][["#TradeId", "NettingSetId", "BaselEEPE"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
