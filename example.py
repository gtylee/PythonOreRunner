import os
from pathlib import Path

from py_ore_tools import OreBasic
from py_ore_tools.repo_paths import default_ore_bin, require_engine_repo_root


# configure your own ore runner as needed
class MyOreExampleRunner(OreBasic):

    def all_trades_processed(self):
        self.parse_input()
        self.parse_output()
        trade_ids_input = [t.attrib['id'] for t in self.input.xml['portfolio'].getroot().findall('./Trade')]
        trade_ids_output = list(self.output.csv['npv']['#TradeId'])
        return sorted(trade_ids_input) == sorted(trade_ids_output)


# setup your folders (the Examples in this case)
engine_root = require_engine_repo_root()
my_example_folder = engine_root / "Examples" / "Legacy" / "Example_1"
ore_exe = Path(os.environ.get("ORE_EXE", str(default_ore_bin())))

# attach ore config folders to Python object
my_ore = MyOreExampleRunner.from_folders(
    input_folder=str(my_example_folder / "Input"),
    output_folder=str(my_example_folder / "Output"),
    execution_folder=str(my_example_folder)
)

# kick off a run
my_ore.run(ore_exe=str(ore_exe), delete_output_folder_before_run=False)

# inspect the output files
my_ore.parse_output()
print("Run produced output files: " + str(len(my_ore.output.locations)))

# extract data from the output file
print(my_ore.output.csv['npv'][['#TradeId', 'TradeType', 'NPV(Base)']])
print(my_ore.output.csv['xva'][['#TradeId', 'NettingSetId', 'BaselEEPE']])

# plot every nettingset_exposure* and every trade_exposure*
try:
    my_ore.plots.plot_nettingset_exposures()
    my_ore.plots.plot_trade_exposures()
except Exception as exc:
    print("Skipping plots due to plotting backend/data issue:", exc)

# do some validation
print("All trades processed: " + str(my_ore.all_trades_processed()))

# backup current run to some safe location
archiv_folder = Path(__file__).resolve().parent / "parity_artifacts" / "ore_archive"
current_run = "attempt_01"
my_ore.backup_inputfiles_to(str(archiv_folder / current_run / "Input"))
my_ore.backup_outputfolder_to(str(archiv_folder / current_run / "Output"))
