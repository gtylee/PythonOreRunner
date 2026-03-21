from __future__ import annotations

from pprint import pformat

from pythonore.payoff_ir.ir import PayoffModuleIR
from pythonore.payoff_ir.normalize import normalize_module


def render_module(module: PayoffModuleIR) -> str:
    return pformat(normalize_module(module))


def diff_modules(left: PayoffModuleIR, right: PayoffModuleIR) -> str:
    ltxt = render_module(left).splitlines()
    rtxt = render_module(right).splitlines()
    out = []
    max_len = max(len(ltxt), len(rtxt))
    for i in range(max_len):
        lv = ltxt[i] if i < len(ltxt) else ""
        rv = rtxt[i] if i < len(rtxt) else ""
        if lv != rv:
            out.append(f"- {lv}")
            out.append(f"+ {rv}")
    return "\n".join(out)
