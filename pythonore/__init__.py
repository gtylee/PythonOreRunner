from __future__ import annotations

from pathlib import Path


_PKG_DIR = Path(__file__).resolve().parent
_SRC_IMPL = _PKG_DIR.parent / "src" / "pythonore"

__path__ = [str(_PKG_DIR)]
if _SRC_IMPL.exists():
    __path__.append(str(_SRC_IMPL))

from pythonore.domain import *  # noqa: F401,F403,E402
from pythonore.io import *  # noqa: F401,F403,E402
from pythonore.mapping import *  # noqa: F401,F403,E402
from pythonore.runtime import *  # noqa: F401,F403,E402
from pythonore.workflows import *  # noqa: F401,F403,E402
