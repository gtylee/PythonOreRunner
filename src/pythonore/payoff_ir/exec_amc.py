from __future__ import annotations

from pythonore.payoff_ir.exec_numpy import NumpyExecutionEnv, execute_numpy
from pythonore.payoff_ir.ir import PayoffModuleIR


def execute_amc(module: PayoffModuleIR, env: NumpyExecutionEnv):
    """AMC execution backend hook.

    The canonical IR exposes continuation semantically via ContinuationValueExpr.
    This backend currently delegates expression evaluation to the NumPy executor
    and expects the caller to provide an env.continuation callback implementing
    regression / backward induction policy.
    """

    return execute_numpy(module, env)
