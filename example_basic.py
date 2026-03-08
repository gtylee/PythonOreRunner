import os
from pathlib import Path

from py_ore_tools import OreBasic

# Folders: set ORE_EXAMPLE_DIR (Input folder or parent) and ORE_EXE, or use ORE repo layout
_project_root = Path(__file__).resolve().parent
_ore_repo_root = _project_root.parent.parent
my_example_folder = Path(os.environ.get("ORE_EXAMPLE_DIR", str(_ore_repo_root / "Examples" / "Example_7")))
ore_exe = Path(os.environ.get("ORE_EXE", str(_ore_repo_root / "build" / "apple-make-relwithdebinfo-arm64" / "App" / "ore")))

# attach ore config folders to Python object
my_ore = OreBasic.from_folders(
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
