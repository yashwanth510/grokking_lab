from dataclasses import replace

import pytest
import torch

from grokking_lab import core


def small():
    return core.Config(
        p=7, d_model=16, n_heads=2, d_mlp=24, steps=4, log_every=2, checkpoint_every=2
    )


def test_nested_disjoint_splits_and_labels():
    c = small()
    x, y, s = core.make_data(c)
    assert torch.equal(y, (x[:, 0] + x[:, 1]) % c.p)
    _, _, larger = core.make_data(replace(c, train_fraction=0.5))
    assert torch.equal(s["train"], larger["train"][: len(s["train"])])
    assert not set(s["train"].tolist()) & set(s["val"].tolist())
    assert not set(s["train"].tolist()) & set(s["test"].tolist())


def test_causal_attention_and_fft_round_trip():
    c = small()
    m = core.TinyTransformer(c).to(core.DEVICE)
    x, _, _ = core.make_data(c)
    logits, cache = m(x.to(core.DEVICE), return_cache=True)
    assert cache["attention"][:, :, 0, 1:].count_nonzero() == 0
    changed, retained = core.project_embeddings(m, list(range(c.p // 2 + 1)))
    assert abs(retained - 1) < 1e-5
    assert torch.allclose(logits, changed(x.to(core.DEVICE)), atol=1e-6)


def test_training_resume_and_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "ROOT", tmp_path)
    c = small()
    m, h, folder = core.run(c, "run")
    loaded, _ = core.load_snapshot(folder, c.steps)
    x, _, _ = core.make_data(c)
    assert torch.equal(m(x.to(core.DEVICE)), loaded(x.to(core.DEVICE)))
    _, resumed, _ = core.run(c, "run", resume=True)
    assert resumed.step.tolist() == h.step.tolist()
    with pytest.raises(FileExistsError):
        core.run(c, "run")
    with pytest.raises(AssertionError):
        core.run(replace(c, seed=99), "run", resume=True)


def test_interrupted_training_matches_uninterrupted(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "ROOT", tmp_path)
    monkeypatch.setattr(core, "DEVICE", torch.device("cpu"))
    c = replace(small(), steps=6)
    uninterrupted, expected_history, _ = core.run(c, "uninterrupted")
    _, partial, _ = core.run(c, "interrupted", stop_at=2)
    assert partial.step.tolist() == [0, 2]
    resumed, actual_history, _ = core.run(c, "interrupted", resume=True)
    for key, value in uninterrupted.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[key]), key
    columns = [c for c in expected_history.columns if c != "elapsed_seconds"]
    assert expected_history[columns].equals(actual_history[columns])


def test_extension_preserves_parent_and_matches_long_run(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(core, "ROOT", tmp_path)
    c = small()
    _, _, source = core.run(c, "parent")
    before = (source / "latest.pt").read_bytes()
    extended, _, folder = core.extend_run(source, "longer", steps=8)
    direct, _, _ = core.run(replace(c, steps=8), "direct")
    assert before == (source / "latest.pt").read_bytes()
    assert json.loads((folder / "extension.json").read_text())["new_steps"] == 8
    for key, value in direct.state_dict().items():
        assert torch.equal(value, extended.state_dict()[key]), key
    with pytest.raises(FileExistsError):
        core.extend_run(source, "longer", steps=8)


def test_reference_settings_and_probe_artifact(tmp_path, monkeypatch):

    monkeypatch.setattr(core, "ROOT", tmp_path)
    c = replace(
        small(), init_std=0.8 / (16**0.5), train_mlp_bias=False, loss_dtype="float64"
    )
    model, _, folder = core.run(c, "reference")
    assert not model.fc1.bias.requires_grad
    assert not model.fc2.bias.requires_grad
    assert torch.count_nonzero(model.fc1.bias) == 0
    x, y, split = core.make_data(c)
    table = core.save_probes(model, x, y, split, folder, c.steps, frequencies=(1, 2))
    assert table.frequency.tolist() == [1, 2]
    assert set(
        ["checkpoint_step", "alpha", "r2", "shuffled_r2", "shuffle_seed"]
    ).issubset(table.columns)
    assert (folder / "probes_validation.csv").is_file()
    assert core.loss_fn(torch.randn(3, c.p), y[:3], c).dtype == torch.float64
