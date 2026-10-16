import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import copy

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path("runs")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(2)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@dataclass(frozen=True)
class Config:
    p: int = 113
    d_model: int = 128
    n_heads: int = 4
    d_mlp: int = 512
    train_fraction: float = 0.30
    split_seed: int = 2026
    seed: int = 0
    lr: float = 1e-3
    weight_decay: float = 1.0
    steps: int = 30000
    log_every: int = 100
    checkpoint_every: int = 1000


def make_data(c):
    a, b = torch.meshgrid(torch.arange(c.p), torch.arange(c.p), indexing="ij")
    x = torch.stack([a.flatten(), b.flatten(), torch.full((c.p * c.p,), c.p)], dim=1)
    y = (x[:, 0] + x[:, 1]) % c.p
    g = torch.Generator().manual_seed(c.split_seed)
    order = torch.randperm(len(y), generator=g)
    n_holdout = int(0.20 * len(y))
    val = order[:n_holdout]
    test = order[n_holdout : 2 * n_holdout]
    pool = order[2 * n_holdout :]
    n_train = int(c.train_fraction * len(y))
    assert 0 < n_train <= len(pool)
    train = pool[:n_train]
    assert not set(train.tolist()) & set(val.tolist())
    assert not set(train.tolist()) & set(test.tolist())
    assert not set(val.tolist()) & set(test.tolist())
    return x, y, {"train": train, "val": val, "test": test, "pool": pool}


class TinyTransformer(nn.Module):
    def __init__(self, c):
        super().__init__()
        assert c.d_model % c.n_heads == 0
        self.c = c
        self.token = nn.Embedding(c.p + 1, c.d_model)
        self.pos = nn.Parameter(torch.empty(3, c.d_model))
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        self.out = nn.Linear(c.d_model, c.d_model, bias=False)
        self.fc1 = nn.Linear(c.d_model, c.d_mlp)
        self.fc2 = nn.Linear(c.d_mlp, c.d_model)
        self.unembed = nn.Linear(c.d_model, c.p, bias=False)
        self.register_buffer(
            "causal", torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
        )
        self.apply(self.initialize)
        nn.init.normal_(self.pos, std=0.02)

    @staticmethod
    def initialize(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, tokens, ablate_heads=(), return_cache=False):
        B, T = tokens.shape
        H, D = self.c.n_heads, self.c.d_model // self.c.n_heads
        embedded = self.token(tokens) + self.pos[:T]
        qkv = self.qkv(embedded).reshape(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(D)
        scores = scores.masked_fill(self.causal[:T, :T], float("-inf"))
        attention = scores.softmax(dim=-1)
        heads = attention @ v
        if ablate_heads:
            mask = torch.ones(H, device=tokens.device)
            mask[list(ablate_heads)] = 0
            heads = heads * mask[None, :, None, None]
        merged = heads.transpose(1, 2).reshape(B, T, H * D)
        resid = embedded + self.out(merged)
        mlp_hidden = F.relu(self.fc1(resid))
        final = resid + self.fc2(mlp_hidden)
        logits = self.unembed(final[:, -1])
        if return_cache:
            return logits, {
                "attention": attention.detach(),
                "mlp_last": mlp_hidden[:, -1].detach(),
                "final_last": final[:, -1].detach(),
            }
        return logits


@torch.no_grad()
def evaluate(model, x_cpu, y_cpu, ids, ablate_heads=(), batch_size=1024):
    model.eval()
    loss_sum = 0.0
    correct = 0
    for chunk in ids.split(batch_size):
        xx, yy = x_cpu[chunk].to(DEVICE), y_cpu[chunk].to(DEVICE)
        logits = model(xx, ablate_heads=ablate_heads)
        loss_sum += F.cross_entropy(logits, yy, reduction="sum").item()
        correct += (logits.argmax(-1) == yy).sum().item()
    return {"loss": loss_sum / len(ids), "accuracy": correct / len(ids)}


def run(c, name, resume=False):
    validate_config(c)
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("Run name must be a single directory name")
    ROOT.mkdir(parents=True, exist_ok=True)
    folder = ROOT / name
    folder.mkdir(exist_ok=True)
    if any(folder.iterdir()) and not resume:
        raise FileExistsError(
            f"{folder} already has a run. Choose another name or resume=True."
        )
    seed_all(c.seed)
    xx, yy, ss = make_data(c)
    m = TinyTransformer(c).to(DEVICE)
    optimizer = torch.optim.AdamW(
        m.parameters(), lr=c.lr, betas=(0.9, 0.98), weight_decay=c.weight_decay
    )
    history, start, previous_seconds = [], 0, 0.0
    if resume:
        # Load only checkpoints you created and trust.
        ck = torch.load(folder / "latest.pt", map_location=DEVICE, weights_only=False)
        assert ck["config"] == asdict(c), "Resume with the same configuration."
        m.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        history, start = ck["history"], ck["step"]
        previous_seconds = ck["elapsed_seconds"]
    (folder / "config.json").write_text(json.dumps(asdict(c), indent=2))
    np.savez(folder / "split_indices.npz", **{k: v.numpy() for k, v in ss.items()})
    train_x, train_y = xx[ss["train"]].to(DEVICE), yy[ss["train"]].to(DEVICE)
    t0 = time.perf_counter()

    def elapsed():
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        return previous_seconds + time.perf_counter() - t0

    def log(step):
        tr = evaluate(m, xx, yy, ss["train"])
        va = evaluate(m, xx, yy, ss["val"])
        norm = sum(w.detach().square().sum().item() for w in m.parameters()) ** 0.5
        row = {
            "step": step,
            "train_loss": tr["loss"],
            "val_loss": va["loss"],
            "train_acc": tr["accuracy"],
            "val_acc": va["accuracy"],
            "weight_norm": norm,
            "elapsed_seconds": elapsed(),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})

    def save(step, snapshot=False):
        checkpoint = {
            "config": asdict(c),
            "model": m.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "history": history,
            "elapsed_seconds": elapsed(),
        }
        tmp = folder / "temporary.pt"
        torch.save(checkpoint, tmp)
        tmp.replace(folder / "latest.pt")
        if snapshot:
            torch.save(
                {"config": asdict(c), "model": m.state_dict(), "step": step},
                folder / f"model_step_{step:06d}.pt",
            )

    if start == 0:
        log(0)
        save(0, snapshot=True)
    for step in range(start + 1, c.steps + 1):
        m.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(m(train_x), train_y)
        if not torch.isfinite(loss):
            raise RuntimeError(
                "Nonfinite loss: inspect learning rate and initialization."
            )
        loss.backward()
        optimizer.step()
        if step % c.log_every == 0 or step == c.steps:
            log(step)
        if step % c.checkpoint_every == 0 or step == c.steps:
            save(step, snapshot=True)
    return m, pd.DataFrame(history), folder


def plot_history(h, folder):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(h.step, h.train_loss, label="train")
    ax[0].plot(h.step, h.val_loss, label="validation")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Cross-entropy")
    ax[1].plot(h.step, h.train_acc, label="train")
    ax[1].plot(h.step, h.val_acc, label="validation")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_ylim(-0.02, 1.02)
    ax[2].plot(h.step, h.weight_norm)
    ax[2].set_ylabel("Parameter L2 norm")
    for a in ax:
        a.set_xlabel("Optimizer updates")
        a.grid(alpha=0.2)
    ax[0].legend()
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(folder / "learning_curves.png", dpi=160)
    plt.close(fig)


def first_sustained(h, column, threshold=0.99, window=3):
    values = h[column].to_numpy() >= threshold
    for i in range(len(values) - window + 1):
        if values[i : i + window].all():
            return int(h.iloc[i].step)
    return None


def summarize(h):
    t = first_sustained(h, "train_acc")
    v = first_sustained(h, "val_acc")
    return {
        "train_crossing": t,
        "validation_crossing": v,
        "delay_updates": v - t if t is not None and v is not None else None,
        "last_validation_accuracy": float(h.iloc[-1].val_acc),
        "elapsed_seconds": float(h.iloc[-1].elapsed_seconds),
    }


def load_snapshot(folder, step):
    ck = torch.load(
        folder / f"model_step_{step:06d}.pt", map_location="cpu", weights_only=True
    )
    c = Config(**ck["config"])
    m = TinyTransformer(c).to(DEVICE)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, c


@torch.no_grad()
def embedding_spectrum(m):
    embedding = m.token.weight[: m.c.p].detach().float().cpu()
    fourier = torch.fft.rfft(embedding, dim=0, norm="ortho")
    return fourier.abs().square().sum(dim=1)


@torch.no_grad()
def collect_mlp(m, xx, batch_size=512):
    pieces = []
    for batch in xx.split(batch_size):
        _, c = m(batch.to(DEVICE), return_cache=True)
        pieces.append(c["mlp_last"].cpu())
    return torch.cat(pieces).double()


def probe_frequency(m, xx, yy, ss, k=1, alpha=1e-2, shuffle=False):
    hidden = collect_mlp(m, xx)
    H = torch.cat([hidden, torch.ones(len(hidden), 1, dtype=torch.double)], dim=1)
    angle = 2 * math.pi * k * yy.double() / m.c.p
    targets = torch.stack([torch.cos(angle), torch.sin(angle)], dim=1)
    tr, va = ss["train"], ss["val"]
    train_targets = targets[tr]
    if shuffle:
        g = torch.Generator().manual_seed(123)
        train_targets = train_targets[torch.randperm(len(tr), generator=g)]
    A = H[tr]
    W = torch.linalg.solve(
        A.T @ A + alpha * torch.eye(A.shape[1], dtype=A.dtype), A.T @ train_targets
    )
    pred = H[va] @ W
    sse = (pred - targets[va]).square().sum()
    sst = (targets[va] - targets[va].mean(0)).square().sum()
    return float(1 - sse / sst)


@torch.no_grad()
def project_embeddings(m, keep):
    changed = copy.deepcopy(m)
    E = m.token.weight[: m.c.p].detach()
    spectrum = torch.fft.rfft(E.float(), dim=0, norm="ortho")
    mask = torch.zeros(spectrum.shape[0], device=E.device)
    mask[list(keep)] = 1
    projected = torch.fft.irfft(spectrum * mask[:, None], n=m.c.p, dim=0, norm="ortho")
    changed.token.weight[: m.c.p].copy_(projected.to(E.dtype))
    retained = float(projected.square().sum() / E.square().sum())
    return changed, retained


def validate_config(c):
    if c.p < 5 or c.d_model < 1 or c.n_heads < 1 or c.d_model % c.n_heads:
        raise ValueError("Require p >= 5 and positive model width divisible by heads")
    if c.d_mlp < 1 or c.steps < 1 or c.log_every < 1 or c.checkpoint_every < 1:
        raise ValueError("Widths, steps and logging intervals must be positive")
    if not 0 < c.train_fraction <= 0.6 or int(c.train_fraction * c.p * c.p) < 1:
        raise ValueError(
            "Training fraction must select a nonempty subset of the 60% pool"
        )
    if (
        not math.isfinite(c.lr)
        or c.lr <= 0
        or not math.isfinite(c.weight_decay)
        or c.weight_decay < 0
    ):
        raise ValueError("Learning rate must be positive; decay must be nonnegative")
