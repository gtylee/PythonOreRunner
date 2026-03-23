import os
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent / "src"
if _SRC_ROOT.exists():
    src_root_text = str(_SRC_ROOT)
    if src_root_text not in sys.path:
        sys.path.insert(0, src_root_text)

from pythonore.repo_paths import find_engine_repo_root, find_ore_bin, local_parity_artifacts_root


def main() -> int:
    engine_root = find_engine_repo_root()
    ore_bin = find_ore_bin()
    if engine_root is None or ore_bin is None:
        print("example.py requires a local ORE Engine checkout and built ore binary.")
        print("Set ENGINE_REPO_ROOT and/or ORE_EXE, or use the Python-only snapshot examples instead.")
        return 0

    from pythonore.ore import OreBasic

    class MyOreExampleRunner(OreBasic):

        def all_trades_processed(self):
            self.parse_input()
            self.parse_output()
            trade_ids_input = [t.attrib["id"] for t in self.input.xml["portfolio"].getroot().findall("./Trade")]
            trade_ids_output = list(self.output.csv["npv"]["#TradeId"])
            return sorted(trade_ids_input) == sorted(trade_ids_output)

    example_dir = Path(os.environ.get("ORE_EXAMPLE_DIR", str(engine_root / "Examples" / "Legacy" / "Example_1")))
    my_ore = MyOreExampleRunner.from_folders(
        input_folder=str(example_dir / "Input"),
        output_folder=str(example_dir / "Output"),
        execution_folder=str(example_dir),
    )
    my_ore.run(ore_exe=str(ore_bin), delete_output_folder_before_run=False)
    my_ore.parse_output()

    print("Run produced output files: " + str(len(my_ore.output.locations)))
    print(my_ore.output.csv["npv"][["#TradeId", "TradeType", "NPV(Base)"]])
    print(my_ore.output.csv["xva"][["#TradeId", "NettingSetId", "BaselEEPE"]])

    my_ore.plots.plot_nettingset_exposures()
    my_ore.plots.plot_trade_exposures()
    print("All trades processed: " + str(my_ore.all_trades_processed()))

    archive_folder = local_parity_artifacts_root() / "ore_archive"
    current_run = "attempt_01"
    my_ore.backup_inputfiles_to(str(archive_folder / current_run / "Input"))
    my_ore.backup_outputfolder_to(str(archive_folder / current_run / "Output"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
