"""Execute notebooks in the repo to validate the project notebook stack."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def _run_notebook(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(notebook, timeout=600, kernel_name="python3")
    client.execute()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all HiSE notebooks to validate the stack")
    parser.add_argument("paths", nargs="*", help="Notebook paths to execute")
    args = parser.parse_args()

    file_path = Path(__file__).resolve()
    root = file_path.parents[2] if len(file_path.parents) > 2 else file_path.parents[1]
    notebook_paths = [Path(item) for item in args.paths] if args.paths else sorted((root / "notebooks").glob("*.ipynb"))

    for path in notebook_paths:
        print(f"Running {path}")
        try:
            _run_notebook(path)
        except Exception as exc:  # pragma: no cover - validation script
            print(f"FAILED: {path}")
            traceback.print_exc()
            raise SystemExit(f"Notebook validation failed for {path}: {exc}") from exc
    print(f"Validated {len(notebook_paths)} notebook(s).")


if __name__ == "__main__":
    main()
