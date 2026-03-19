import os
from pathlib import Path

from py_ore_tools import OreBasic
from py_ore_tools.repo_paths import default_ore_bin, require_engine_repo_root

# setup your folders (the Examples in this case)
engine_root = require_engine_repo_root()
my_example_folder = Path(os.environ.get("ORE_EXAMPLE_DIR", str(engine_root / "Examples" / "Legacy" / "Example_1")))
ore_exe = Path(os.environ.get("ORE_EXE", str(default_ore_bin())))

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
