"""Synchronize self-contained notebook definitions with the package (no downloads)."""

import argparse
import ast
import json
import pprint
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/Grokking_Research_Lab.ipynb"


def definitions(source):
    lines = source.splitlines()
    result = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            result[node.name] = "\n".join(lines[start - 1 : node.end_lineno])
    return result


def synchronize():
    package = definitions((ROOT / "src/grokking_lab/core.py").read_text())
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    reference = json.loads((ROOT / "configs/reference.json").read_text())
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        lines = cell.source.splitlines()
        replacements = []
        for node in ast.parse(cell.source).body:
            if (
                isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in package
            ):
                start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                replacements.append((start - 1, node.end_lineno, package[node.name]))
            elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "cfg" for t in node.targets
            ):
                replacements.append(
                    (
                        node.lineno - 1,
                        node.end_lineno,
                        "cfg = Config(**"
                        + pprint.pformat(reference, sort_dicts=False)
                        + ")",
                    )
                )
        for start, end, text in sorted(replacements, reverse=True):
            lines[start:end] = text.splitlines()
        cell.source = "\n".join(lines)
        names = definitions(cell.source)
        if "run" in names:
            for name in ["loss_fn", "validate_config"]:
                if name not in names:
                    cell.source = package[name] + "\n\n\n" + cell.source
        if "probe_frequency" in names and "save_probes" not in names:
            cell.source += "\n\n\n" + package["save_probes"]
    nbformat.validate(notebook)
    return nbformat.writes(notebook)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    updated = synchronize()
    if args.check:
        if NOTEBOOK.read_text() != updated:
            raise SystemExit(
                "Notebook is out of sync. Run python scripts/sync_notebook.py"
            )
        print("Notebook and package definitions are synchronized.")
    else:
        NOTEBOOK.write_text(updated)
        print("Updated", NOTEBOOK.relative_to(ROOT))
