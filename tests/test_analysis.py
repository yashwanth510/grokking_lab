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


def test_cli_merges_notebook_checkpoint_summary(tmp_path, monkeypatch):
    import sys

    from grokking_lab import __main__ as cli

    folder = tmp_path / "run" / "analysis"
    folder.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "step": 0,
                "train_accuracy": 0.1,
                "validation_accuracy": 0.2,
                "frequency_1_r2": -0.1,
                "shuffled_frequency_1_r2": -0.2,
            },
            {
                "step": 1000,
                "train_accuracy": 0.9,
                "validation_accuracy": 0.8,
                "frequency_1_r2": 0.3,
                "shuffled_frequency_1_r2": -0.1,
            },
        ]
    ).to_csv(folder / "checkpoint_summary.csv", index=False)
    monkeypatch.setattr(
        cli,
        "analyze",
        lambda *_: {
            "checkpoint_step": 1000,
            "train": {"accuracy": 1.0},
            "validation": {"accuracy": 0.95},
            "probe_frequency_1_r2": 0.5,
            "shuffled_probe_r2": -0.05,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "grokking_lab",
            "analyze",
            "--output",
            str(tmp_path),
            "--name",
            "run",
            "--checkpoints",
            "1000",
        ],
    )
    cli.main()
    table = pd.read_csv(folder / "checkpoint_summary.csv")
    assert table.checkpoint_step.tolist() == [0, 1000]
    assert table.validation_accuracy.tolist() == [0.2, 0.95]
    assert table.probe_frequency_1_r2.tolist() == [-0.1, 0.5]
    assert table.shuffled_probe_r2.tolist() == [-0.2, -0.05]
    assert table.isna().sum().sum() == 0
