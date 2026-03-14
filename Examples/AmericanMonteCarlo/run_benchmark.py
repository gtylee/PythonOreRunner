#!/usr/bin/env python

import glob
import os
import sys
sys.path.append('../')
from ore_examples_helper import OreExample

oreex = OreExample(sys.argv[1] if len(sys.argv)>1 else False)

print("+-----------------------------------------------------+")
print("| AMC-CG Benchmark                                    |")
print("+-----------------------------------------------------+")

oreex.print_headline("Run ORE to produce AMC-CG exposure")
oreex.run("Input/ore_amccg.xml")

oreex.print_headline("Plot results: AMC-CG simulated exposure")

oreex.setup_plot("amccg_benchmark_bermudanswaption")
oreex.plot("amccg/exposure_trade_BermSwp.csv", 2, 3, 'b', "AMC-CG Swaption EPE")
oreex.plot("amccg/exposure_trade_BermSwp.csv", 2, 4, 'r', "AMC-CG Swaption ENE")
oreex.decorate_plot(title="AMC-CG benchmark exposure for 10y10y EUR Bermudan Payer Swaption")
oreex.save_plot_to_file()

oreex.setup_plot("amccg_benchmark_vanillaswap_eur")
oreex.plot("amccg/exposure_trade_Swap_EUR.csv", 2, 3, 'b', "AMC-CG Vanilla Swap EUR EPE")
oreex.plot("amccg/exposure_trade_Swap_EUR.csv", 2, 4, 'r', "AMC-CG Vanilla Swap EUR ENE")
oreex.decorate_plot(title="AMC-CG benchmark exposure for 20y EUR Payer Swap")
oreex.save_plot_to_file()

oreex.setup_plot("amccg_benchmark_vanillaswap_usd")
oreex.plot("amccg/exposure_trade_Swap_USD.csv", 2, 3, 'b', "AMC-CG Vanilla Swap USD EPE")
oreex.plot("amccg/exposure_trade_Swap_USD.csv", 2, 4, 'r', "AMC-CG Vanilla Swap USD ENE")
oreex.decorate_plot(title="AMC-CG benchmark exposure for 20y USD Payer Swap")
oreex.save_plot_to_file()

oreex.setup_plot("amccg_benchmark_fxoption")
oreex.plot("amccg/exposure_trade_FX_CALL_OPTION.csv", 2, 3, 'b', "AMC-CG FX Call Option EPE")
oreex.plot("amccg/exposure_trade_FX_CALL_OPTION.csv", 2, 4, 'r', "AMC-CG FX Call Option ENE")
oreex.decorate_plot(title="AMC-CG benchmark exposure for FX Call Option EUR-USD")
oreex.save_plot_to_file()

oreex.setup_plot("amccg_benchmark_xccyswap")
oreex.plot("amccg/exposure_trade_CC_SWAP_EUR_USD.csv", 2, 3, 'b', "AMC-CG XCcy Swap EPE")
oreex.plot("amccg/exposure_trade_CC_SWAP_EUR_USD.csv", 2, 4, 'r', "AMC-CG XCcy Swap ENE")
oreex.decorate_plot(title="AMC-CG benchmark exposure for XCcy Swap EUR-USD")
oreex.save_plot_to_file()

oreex.setup_plot("amccg_benchmark_nettingset")
oreex.plot("amccg/exposure_nettingset_CPTY_A.csv", 2, 3, 'b', "AMC-CG Netting Set EPE")
oreex.plot("amccg/exposure_nettingset_CPTY_A.csv", 2, 4, 'r', "AMC-CG Netting Set ENE")
oreex.decorate_plot(title="AMC-CG benchmark exposure for the netting set")
oreex.save_plot_to_file()
