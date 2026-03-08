from __future__ import annotations

import sys
from pathlib import Path


# Allow `import native_xva_interface` when pytest is launched from the repo root.
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
