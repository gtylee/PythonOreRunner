import os
from pathlib import Path

from py_ore_tools import OreBasic

# setup your folders (the Examples in this case)
repo_root = Path(__file__).resolve().parents[2]
my_example_folder = repo_root / "Examples" / "Example_7"
ore_exe = Path(os.environ.get("ORE_EXE", str(repo_root / "build" / "apple-make-relwithdebinfo-arm64" / "App" / "ore")))

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
