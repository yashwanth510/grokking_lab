"""Train, resume, and analyze local, Colab, or Kaggle experiments."""

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from . import core


def analyze(folder, step=None, checkpoint_dir=None):
    config = core.Config(**json.loads((folder / "config.json").read_text()))
    step = config.steps if step is None else step
    checkpoint_dir = folder if checkpoint_dir is None else checkpoint_dir
    model, config = core.load_snapshot(checkpoint_dir, step)
    x, y, split = core.make_data(config)
    history = pd.read_csv(folder / "history.csv")
    history = history[history.step <= step]
    output = folder / "analysis" / f"step_{step:06d}"
    output.mkdir(parents=True, exist_ok=True)
    core.plot_history(history, output)
    energy = core.embedding_spectrum(model)
    pd.DataFrame({"frequency": range(len(energy)), "energy": energy.numpy()}).to_csv(
        output / "embedding_spectrum.csv", index=False
    )
    baseline = core.evaluate(model, x, y, split["val"])
    rows = [{"intervention": "none", **baseline}]
    for heads in [(h,) for h in range(config.n_heads)] + [tuple(range(config.n_heads))]:
        rows.append(
            {
                "intervention": str(heads),
                **core.evaluate(model, x, y, split["val"], ablate_heads=heads),
            }
        )
    pd.DataFrame(rows).to_csv(output / "head_ablations_validation.csv", index=False)
    top = (torch.topk(energy[1:], min(5, len(energy) - 1)).indices + 1).tolist()
    sets = {
        "all": list(range(len(energy))),
        "keep_top": [0] + top,
        "remove_top": [k for k in range(len(energy)) if k not in top],
    }
    generator = torch.Generator().manual_seed(999)
    for trial in range(5):
        sets[f"keep_random_{trial}"] = [0] + (
            torch.randperm(len(energy) - 1, generator=generator)[: len(top)] + 1
        ).tolist()
    rows = []
    for name, keep in sets.items():
        changed, retained = core.project_embeddings(model, keep)
        metrics = core.evaluate(changed, x, y, split["val"])
        if name == "all" and abs(metrics["loss"] - baseline["loss"]) >= 1e-4:
            raise RuntimeError("FFT round-trip control failed")
        rows.append(
            {
                "condition": name,
                "kept_frequencies": keep,
                "retained_energy": retained,
                **metrics,
            }
        )
    pd.DataFrame(rows).to_csv(
        output / "frequency_interventions_validation.csv", index=False
    )
    probes = core.save_probes(model, x, y, split, output, step)
    summary = {
        "checkpoint_step": step,
        "checkpoint_sha256": hashlib.sha256(
            (checkpoint_dir / f"model_step_{step:06d}.pt").read_bytes()
        ).hexdigest(),
        "config": asdict(config),
        "analysis_device": str(core.DEVICE),
        "analysis_torch": str(torch.__version__),
        "train": core.evaluate(model, x, y, split["train"]),
        "validation": baseline,
        **core.summarize(history),
        "probe_frequency_1_r2": float(probes.iloc[0].r2),
        "shuffled_probe_r2": float(probes.iloc[0].shuffled_r2),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "analyze", "test", "extend"])
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.json"))
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--name", default="smoke_seed0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--stop-at",
        type=int,
        help="Checkpoint and stop early without changing the declared horizon",
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Original run directory for an exploratory extension",
    )
    parser.add_argument(
        "--steps", type=int, help="New horizon for an exploratory extension"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Read model snapshots from a separate backup directory",
    )
    snapshots = parser.add_mutually_exclusive_group()
    snapshots.add_argument(
        "--checkpoints", type=int, nargs="+", help="Saved steps to analyze"
    )
    snapshots.add_argument("--all-checkpoints", action="store_true")
    parser.add_argument(
        "--protocol-frozen",
        action="store_true",
        help="Confirm analysis choices were frozen before test access",
    )
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("Run name must be a single directory name")
    core.ROOT = args.output
    folder = args.output / args.name
    if args.command == "train":
        config = core.Config(**json.loads(args.config.read_text()))
        core.validate_config(config)
        args.output.mkdir(parents=True, exist_ok=True)
        env = {
            "python": sys.version,
            "torch": str(torch.__version__),
            "platform": platform.platform(),
            "device": str(core.DEVICE),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        model, history, folder = core.run(
            config, args.name, resume=args.resume, stop_at=args.stop_at
        )
        (folder / "environment.json").write_text(json.dumps(env, indent=2))
        core.plot_history(history, folder)
    elif args.command == "extend":
        if args.source is None or args.steps is None:
            parser.error("extend requires --source and --steps, plus a new --name")
        _, history, folder = core.extend_run(args.source, args.name, args.steps)
        core.plot_history(history, folder)
    elif args.command == "analyze":
        checkpoint_dir = args.checkpoint_dir or folder
        steps = args.checkpoints
        if args.all_checkpoints:
            steps = sorted(
                int(p.stem.removeprefix("model_step_"))
                for p in checkpoint_dir.glob("model_step_*.pt")
            )
            if not steps:
                parser.error("No saved model snapshots found")
        summaries = [
            analyze(folder, step, checkpoint_dir) for step in (steps or [None])
        ]
        rows = [
            {
                "checkpoint_step": row["checkpoint_step"],
                "train_accuracy": row["train"]["accuracy"],
                "validation_accuracy": row["validation"]["accuracy"],
                "probe_frequency_1_r2": row["probe_frequency_1_r2"],
                "shuffled_probe_r2": row["shuffled_probe_r2"],
            }
            for row in summaries
        ]
        table_path = folder / "analysis" / "checkpoint_summary.csv"
        table = pd.DataFrame(rows)
        if table_path.exists():
            existing = pd.read_csv(table_path)
            if "step" in existing and "checkpoint_step" not in existing:
                existing = existing.rename(
                    columns={
                        "step": "checkpoint_step",
                        "frequency_1_r2": "probe_frequency_1_r2",
                        "shuffled_frequency_1_r2": "shuffled_probe_r2",
                    }
                )
            table = pd.concat([existing, table]).drop_duplicates(
                "checkpoint_step", keep="last"
            )
        table.sort_values("checkpoint_step").to_csv(table_path, index=False)
    else:
        if not args.protocol_frozen:
            parser.error(
                "Final test requires --protocol-frozen after freezing analysis choices"
            )
        if (folder / "final_test.json").exists():
            parser.error("Final test already recorded for this run")
        config = core.Config(**json.loads((folder / "config.json").read_text()))
        model, config = core.load_snapshot(folder, config.steps)
        x, y, split = core.make_data(config)
        report = {
            "config": asdict(config),
            "step": config.steps,
            "test": core.evaluate(model, x, y, split["test"]),
        }
        (folder / "final_test.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
