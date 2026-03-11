import io
import unittest
from contextlib import redirect_stdout

import demo_python_curve_fitter as demo


class TestDemoPythonCurveFitter(unittest.TestCase):
    def test_fmt_number_handles_none_float_and_text(self):
        self.assertEqual(demo._fmt_number(None), "-")
        self.assertEqual(demo._fmt_number(""), "-")
        self.assertEqual(demo._fmt_number(1.25), "1.250000000000")
        self.assertEqual(demo._fmt_number("abc"), "abc")

    def test_demo_output_contains_mixed_instrument_sections(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = demo.main(["--points", "2", "--quotes-per-segment", "2"])
        text = buf.getvalue()

        self.assertEqual(rc, 0)
        self.assertIn("Python curve fitter demo", text)
        self.assertIn("Input instruments", text)
        self.assertIn("Deposit | conventions=USD-LIBOR-CONVENTIONS", text)
        self.assertIn("FRA | conventions=USD-3M-FRA-CONVENTIONS", text)
        self.assertIn("Swap | conventions=USD-3M-SWAP-CONVENTIONS", text)
        self.assertIn("First native ORE nodes", text)
        self.assertIn("Fitted outputs on ORE report grid", text)
        self.assertIn("Sample ORE vs Python grid comparison", text)


if __name__ == "__main__":
    unittest.main()
