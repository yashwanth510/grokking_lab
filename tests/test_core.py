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
