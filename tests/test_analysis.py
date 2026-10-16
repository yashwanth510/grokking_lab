import json

import pandas as pd

from grokking_lab import core
from grokking_lab.__main__ import analyze


def test_checkpoint_analysis_preserves_published_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "ROOT", tmp_path)
    c = core.Config(
        p=7, d_model=16, n_heads=2, d_mlp=24, steps=2, log_every=1, checkpoint_every=1
    )
    _, _, folder = core.run(c, "run")
    original = {"source_archive": "original-results.zip"}
    (folder / "summary.json").write_text(json.dumps(original))
    for step in [0, 2]:
        summary = analyze(folder, step)
        output = folder / "analysis" / f"step_{step:06d}"
        table = pd.read_csv(output / "probes_validation.csv")
        assert table.checkpoint_step.unique().tolist() == [step]
        assert summary["checkpoint_step"] == step
        assert len(summary["checkpoint_sha256"]) == 64
    assert json.loads((folder / "summary.json").read_text()) == original
    assert not (folder / "final_test.json").exists()
