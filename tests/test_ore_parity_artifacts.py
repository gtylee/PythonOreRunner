import tempfile
import unittest
from pathlib import Path

from py_ore_tools.ore_parity_artifacts import CaseMetadata, MANDATORY_SUBDIRS, build_case_layout, write_case_manifest


class TestParityArtifacts(unittest.TestCase):
    def test_layout_and_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = build_case_layout(root, "case_a", "fixed")
            for d in MANDATORY_SUBDIRS:
                self.assertTrue((paths.root / d).exists())

            meta = CaseMetadata(
                case_id="case_a",
                run_mode="fixed",
                asof_date="2016-02-05",
                base_ccy="USD",
                model_ccys=("USD",),
                fx_pairs=tuple(),
                indices=("USD-LIBOR-3M",),
                products=("IRS",),
                convention_profile="A",
                ore_samples=2000,
                python_paths=10000,
                seed=42,
            )
            out = write_case_manifest(paths, meta)
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
