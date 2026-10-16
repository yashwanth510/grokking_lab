"""Run the portable notebook's smoke and analysis path without a GPU."""

from pathlib import Path

import nbformat
import pytest
import torch

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/Grokking_Research_Lab.ipynb"


def test_notebook_smoke_analysis_export_and_gpu_guard(tmp_path, monkeypatch):
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    monkeypatch.chdir(tmp_path)
    namespace = {"__name__": "__main__", "display": lambda value: None}
    pilot_source = None
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        exec(compile(cell.source, f"<notebook cell {index}>", "exec"), namespace)
        if "environment =" in cell.source:
            namespace.update(RUN_PILOT=False, SMOKE_STEPS=2, DEVICE=torch.device("cpu"))
        if "class Config:" in cell.source:
            namespace["cfg"] = namespace["replace"](
                namespace["cfg"],
                d_model=16,
                d_mlp=32,
                steps=2,
                log_every=1,
                checkpoint_every=1,
            )
        if "pilot_model, pilot_history, pilot_dir = run(" in cell.source:
            pilot_source = cell.source
    output = tmp_path / "runs/smoke_seed0"
    for name in [
        "history.csv",
        "learning_curves.png",
        "head_ablations_validation.csv",
        "frequency_interventions_validation.csv",
        "model_step_000002.pt",
    ]:
        assert (output / name).is_file()
    assert (tmp_path / "grokking_lab_outputs.zip").is_file()
    assert not (output / "final_test.json").exists()
    assert pilot_source is not None
    namespace["RUN_PILOT"] = True
    with pytest.raises(RuntimeError, match="Enable a GPU"):
        exec(compile(pilot_source, "<pilot cell>", "exec"), namespace)
    namespace["plt"].close("all")
