import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

_SRC_ROOT = Path(__file__).resolve().parent / "src"
if _SRC_ROOT.exists():
    src_root_text = str(_SRC_ROOT)
    if src_root_text not in sys.path:
        sys.path.insert(0, src_root_text)


def _default_fin_system_folder() -> Path:
    repo_root = Path(__file__).resolve().parent
    env_folder = os.environ.get("FIN_SYSTEM_FOLDER", "")
    candidates = [
        Path(env_folder) if env_folder else None,
        repo_root / "ToyExample",
        repo_root / "Tools" / "PythonOreRunner" / "ToyExample",
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "Input").exists():
            return candidate
    return repo_root / "ToyExample"


def main() -> int:
    fin_system_folder = _default_fin_system_folder()
    if not (fin_system_folder / "Input").exists():
        print("example_systemic.py requires a ToyExample-style folder with Input/ and Output/.")
        print("Set FIN_SYSTEM_FOLDER to a prepared case directory before running this script.")
        return 0

    from pythonore.ore import OreBasic

    class FinancialSystem(OreBasic):

        def __init__(self, input_files, output_folder, execution_folder):
            super().__init__(input_files, output_folder, execution_folder)
            self.counterparties = None
            self.graph = None
            self.EEPE = None

        def _set_counterparties(self):
            self.parse_output()
            self.counterparties = sorted(list({c.split(".")[0] for c in self.get_nettingset_eepe().to_dict().keys()}))

        def _set_graph(self):
            self.graph = nx.DiGraph()
            eepe_dict = self.get_nettingset_eepe().to_dict()
            for key in self.get_nettingset_eepe().to_dict().keys():
                self.graph.add_edge(key.split(".")[0], key.split(".")[1], weight=eepe_dict[key])

        def compute_systemic_balancedness(self):
            self.EEPE = np.sum(np.array([e[2]["weight"] for e in self.graph.edges(data=True)]))
            for node_name in self.graph.nodes():
                node = self.graph.nodes[node_name]
                node["EEPE+"] = np.sum(np.array([e[2]["weight"] for e in self.graph.out_edges(node_name, data=True)]))
                node["rho+"] = node["EEPE+"] / self.EEPE
                node["EEPE-"] = np.sum(np.array([e[2]["weight"] for e in self.graph.in_edges(node_name, data=True)]))
                node["rho-"] = node["EEPE-"] / self.EEPE

        def plot(self):
            pos = nx.circular_layout(self.graph)
            nx.draw(self.graph, pos, with_labels=True)
            nx.draw_networkx_edge_labels(self.graph, pos)
            plt.show()

        def get_nettingset_eepe(self):
            return self.output.csv["xva"][self.output.csv["xva"]["#TradeId"].isnull()][["NettingSetId", "BaselEEPE"]].set_index(
                "NettingSetId"
            )["BaselEEPE"]

        def consistent_input(self):
            self.parse_input()
            trades = self.input.xml["portfolio"].getroot().findall("./Trade")
            trade_ids = [t.attrib["id"] for t in trades]
            for k, trade_id in enumerate(trade_ids):
                id_components = trade_id.split(".")
                id_mirror_trade = ".".join([id_components[1], id_components[0], id_components[2]])
                l = trade_ids.index(id_mirror_trade)
                trade_type = trades[k].findall("./TradeType")[0].text
                assert trades[l].findall("./TradeType")[0].text == trade_type
                if trade_type == "FxForward":
                    assert trades[k].findall("./FxForwardData/BoughtCurrency")[0].text == trades[l].findall(
                        "./FxForwardData/SoldCurrency"
                    )[0].text
                    assert trades[k].findall("./FxForwardData/BoughtAmount")[0].text == trades[l].findall(
                        "./FxForwardData/SoldAmount"
                    )[0].text
                elif trade_type == "Swap":
                    to_bool = lambda x: x == "true"
                    assert to_bool(trades[k].findall("./SwapData/LegData/Payer")[0].text) != to_bool(
                        trades[k].findall("./SwapData/LegData/Payer")[1].text
                    )
                    assert to_bool(trades[l].findall("./SwapData/LegData/Payer")[0].text) != to_bool(
                        trades[l].findall("./SwapData/LegData/Payer")[1].text
                    )
                    assert to_bool(trades[k].findall("./SwapData/LegData/Payer")[0].text) != to_bool(
                        trades[l].findall("./SwapData/LegData/Payer")[0].text
                    )
                    assert to_bool(trades[k].findall("./SwapData/LegData/Payer")[1].text) != to_bool(
                        trades[l].findall("./SwapData/LegData/Payer")[1].text
                    )
                else:
                    return False
                netting_set_ids = [ns.text for ns in self.input.xml["netting"].getroot().findall("./NettingSet/NettingSetId")]
                for trade in trades:
                    assert trade.findall("./Envelope/NettingSetId")[0].text in netting_set_ids
            return True

    fs = FinancialSystem.from_folders(
        input_folder=str(fin_system_folder / "Input"),
        output_folder=str(fin_system_folder / "Output"),
        execution_folder=str(fin_system_folder),
    )

    fs.parse_output()
    fs.plots.plot_nettingset_exposures()
    fs.plots.plot_trade_exposures()
    print(fs.output.csv["npv"][["#TradeId", "TradeType", "NPV(Base)"]])
    print(fs.output.csv["xva"][["#TradeId", "NettingSetId", "BaselEEPE"]])
    print(fs.get_nettingset_eepe())

    fs._set_graph()
    fs.compute_systemic_balancedness()
    data = fs.graph.nodes(data=True)
    df = pd.DataFrame(index=[x[0] for x in data], data=[x[1] for x in data])
    print(df)

    fs.get_nettingset_eepe().to_csv(os.path.join(fs.output_folder, "adjacency_matrix.csv"))
    df["rho+"].to_csv(os.path.join(fs.output_folder, "systemic_risk.csv"))
    df["rho+"] = (df["rho+"] * 100).astype(int)
    print("Total EEPE", df["EEPE+"].sum())
    print("Total NPV", fs.output.csv["npv"][["NPV(Base)"]].sum())
    print("Input consistent: " + str(fs.consistent_input()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
