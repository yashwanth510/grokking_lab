"""Train, resume, and analyze local, Colab, or Kaggle experiments."""

import argparse
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from . import core


def analyze(folder):
    config = core.Config(**json.loads((folder / "config.json").read_text()))
    model, config = core.load_snapshot(folder, config.steps)
    x, y, split = core.make_data(config)
    history = pd.read_csv(folder / "history.csv")
    core.plot_history(history, folder)
    energy = core.embedding_spectrum(model)
    pd.DataFrame({"frequency": range(len(energy)), "energy": energy.numpy()}).to_csv(
        folder / "embedding_spectrum.csv", index=False
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
    pd.DataFrame(rows).to_csv(folder / "head_ablations_validation.csv", index=False)
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
        folder / "frequency_interventions_validation.csv", index=False
    )
    summary = {
        **core.summarize(history),
        "probe_frequency_1_r2": core.probe_frequency(model, x, y, split),
        "shuffled_probe_r2": core.probe_frequency(model, x, y, split, shuffle=True),
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "analyze", "test"])
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.json"))
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--name", default="smoke_seed0")
    parser.add_argument("--resume", action="store_true")
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
        model, history, folder = core.run(config, args.name, resume=args.resume)
        (folder / "environment.json").write_text(json.dumps(env, indent=2))
        core.plot_history(history, folder)
    elif args.command == "analyze":
        analyze(folder)
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
