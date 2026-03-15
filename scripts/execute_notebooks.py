from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import traceback
from pathlib import Path
from typing import Any

from jupyter_client import KernelManager


_REPO_ROOT = Path(__file__).resolve().parents[1]
_CACHE_ROOT = _REPO_ROOT / ".cache" / "notebook_exec"
_IPYTHON_DIR = _CACHE_ROOT / "ipython"
_JUPYTER_RUNTIME_DIR = _CACHE_ROOT / "jupyter_runtime"
_MPLCONFIGDIR = _CACHE_ROOT / "mplconfig"
for _path in (_IPYTHON_DIR, _JUPYTER_RUNTIME_DIR, _MPLCONFIGDIR):
    _path.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("IPYTHONDIR", str(_IPYTHON_DIR))
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(_JUPYTER_RUNTIME_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))


def _read_notebook(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_notebook(path: Path, notebook: dict[str, Any]) -> None:
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _serialize_display(content: dict[str, Any], *, output_type: str) -> dict[str, Any]:
    output = {
        "output_type": output_type,
        "data": content.get("data", {}),
        "metadata": content.get("metadata", {}),
    }
    if output_type == "execute_result":
        output["execution_count"] = content.get("execution_count")
    return output


def _execute_notebook(path: Path, *, cwd: Path, timeout: float) -> tuple[int, int]:
    notebook = _read_notebook(path)

    km = KernelManager(kernel_name=notebook.get("metadata", {}).get("kernelspec", {}).get("name", "python3"))
    km.start_kernel(
        cwd=str(cwd),
        env={
            **os.environ,
            "IPYTHONDIR": os.environ["IPYTHONDIR"],
            "JUPYTER_RUNTIME_DIR": os.environ["JUPYTER_RUNTIME_DIR"],
            "MPLCONFIGDIR": os.environ["MPLCONFIGDIR"],
        },
    )
    client = km.blocking_client()
    client.start_channels()
    client.wait_for_ready(timeout=timeout)

    executed = 0
    last_count = 0
    try:
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue

            source = "".join(cell.get("source", []))
            cell["outputs"] = []
            if not source.strip():
                cell["execution_count"] = None
                continue

            msg_id = client.execute(source, store_history=True)
            executed += 1
            while True:
                try:
                    msg = client.get_iopub_msg(timeout=timeout)
                except queue.Empty as exc:
                    raise TimeoutError(f"Timed out while executing {path.name} cell {executed}") from exc

                if msg.get("parent_header", {}).get("msg_id") != msg_id:
                    continue

                msg_type = msg["msg_type"]
                content = msg["content"]

                if msg_type == "status" and content.get("execution_state") == "idle":
                    break
                if msg_type == "execute_input":
                    cell["execution_count"] = content.get("execution_count")
                    last_count = content.get("execution_count") or last_count
                    continue
                if msg_type == "stream":
                    cell["outputs"].append(
                        {
                            "output_type": "stream",
                            "name": content.get("name", "stdout"),
                            "text": content.get("text", ""),
                        }
                    )
                    continue
                if msg_type in {"display_data", "execute_result"}:
                    cell["outputs"].append(_serialize_display(content, output_type=msg_type))
                    if msg_type == "execute_result" and content.get("execution_count") is not None:
                        cell["execution_count"] = content["execution_count"]
                        last_count = content["execution_count"]
                    continue
                if msg_type == "error":
                    cell["outputs"].append(
                        {
                            "output_type": "error",
                            "ename": content.get("ename", ""),
                            "evalue": content.get("evalue", ""),
                            "traceback": content.get("traceback", []),
                        }
                    )
                    _write_notebook(path, notebook)
                    raise RuntimeError(f"{path.name} failed in cell {executed}: {content.get('ename')}: {content.get('evalue')}")
                if msg_type == "clear_output":
                    if content.get("wait"):
                        continue
                    cell["outputs"] = []

            if cell.get("execution_count") is None and last_count:
                cell["execution_count"] = last_count
    finally:
        try:
            client.stop_channels()
        finally:
            km.shutdown_kernel(now=True)

    _write_notebook(path, notebook)
    return executed, last_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute notebooks in place and persist outputs.")
    parser.add_argument("notebooks", nargs="+", help="Notebook paths to execute")
    parser.add_argument("--cwd", default=".", help="Working directory for notebook execution")
    parser.add_argument("--timeout", type=float, default=1800.0, help="Per-cell timeout in seconds")
    args = parser.parse_args(argv)

    cwd = Path(args.cwd).resolve()
    failures = 0
    for raw_path in args.notebooks:
        path = Path(raw_path).resolve()
        print(f"[run] {path}")
        try:
            executed, last_count = _execute_notebook(path, cwd=cwd, timeout=float(args.timeout))
        except Exception as exc:
            failures += 1
            print(f"[fail] {path.name}: {exc}", file=sys.stderr)
            traceback.print_exc()
            continue
        print(f"[ok] {path.name}: executed {executed} code cells, last execution_count={last_count}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
