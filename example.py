import os
from pathlib import Path

from py_ore_tools import OreBasic


# configure your own ore runner as needed
class MyOreExampleRunner(OreBasic):

    def all_trades_processed(self):
        self.parse_input()
        self.parse_output()
        trade_ids_input = [t.attrib['id'] for t in self.input.xml['portfolio'].getroot().findall('./Trade')]
        trade_ids_output = list(self.output.csv['npv']['#TradeId'])
        return sorted(trade_ids_input) == sorted(trade_ids_output)


# Folders: set ORE_EXAMPLE_DIR and ORE_EXE, or use ORE repo layout (see README)
_project_root = Path(__file__).resolve().parent
_ore_repo_root = _project_root.parent.parent
my_example_folder = Path(os.environ.get("ORE_EXAMPLE_DIR", str(_ore_repo_root / "Examples" / "Example_7")))
ore_exe = Path(os.environ.get("ORE_EXE", str(_ore_repo_root / "build" / "apple-make-relwithdebinfo-arm64" / "App" / "ore")))

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
my_ore.plots.plot_nettingset_exposures()
my_ore.plots.plot_trade_exposures()

# do some validation
print("All trades processed: " + str(my_ore.all_trades_processed()))

# backup current run to some safe location
archiv_folder = _project_root / "parity_artifacts" / "ore_archive"
current_run = "attempt_01"
my_ore.backup_inputfiles_to(str(archiv_folder / current_run / "Input"))
my_ore.backup_outputfolder_to(str(archiv_folder / current_run / "Output"))
