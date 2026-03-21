from __future__ import annotations

import ast


_ALLOWED_CTX_CALLS = {
    "number",
    "event",
    "events",
    "index",
    "currency",
    "daycount",
    "where",
    "max",
    "min",
    "pay",
    "logpay",
    "record_result",
    "set_npv",
    "require",
}


class RestrictedPayoffError(ValueError):
    pass


def parse_restricted_payoff(source: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    payoff_funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "payoff"]
    if len(payoff_funcs) != 1:
        raise RestrictedPayoffError("Expected exactly one top-level payoff(ctx) function")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.AsyncFunctionDef, ast.ClassDef, ast.With, ast.Try, ast.While, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp, ast.Lambda, ast.Delete, ast.Global, ast.Nonlocal, ast.Yield, ast.YieldFrom, ast.Await)):
            raise RestrictedPayoffError(f"Unsupported construct {type(node).__name__}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "ctx" and node.func.attr not in _ALLOWED_CTX_CALLS:
                raise RestrictedPayoffError(f"Unsupported ctx method '{node.func.attr}'")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in {"float", "int", "str"}:
                raise RestrictedPayoffError(f"Unsupported function call '{node.func.id}'")
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id != "ctx":
            if node.attr != "at":
                raise RestrictedPayoffError(f"Unsupported attribute access on '{node.value.id}'")
    payoff = payoff_funcs[0]
    if len(payoff.args.args) != 1 or payoff.args.args[0].arg != "ctx":
        raise RestrictedPayoffError("payoff function must take exactly one argument named ctx")
    return payoff
