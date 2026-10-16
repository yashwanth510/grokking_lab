"""Verify the user's backup and prepare or publish the measured pilot release."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "yashwanth510/grokking_lab"
TAG = "pilot-v1"
MEASURED_COMMIT = "35f56f8955dc9e91219b4b8f5b25df72042c9952"


def prepare(backup, output):
    config = json.loads((ROOT / "results/pilot_seed0/config.json").read_text())
    names = ["model_step_000000.pt", "model_step_030000.pt", "latest.pt"]
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(backup) as archive:
        for name in names:
            (output / name).write_bytes(archive.read(name))
    for name in names:
        checkpoint = torch.load(output / name, map_location="cpu", weights_only=True)
        assert all(
            checkpoint["config"][key] == value for key, value in config.items()
        ), "Configuration mismatch"
        assert checkpoint["step"] == (0 if name == "model_step_000000.pt" else 30000)
    final = torch.load(output / names[1], map_location="cpu", weights_only=True)
    latest = torch.load(output / "latest.pt", map_location="cpu", weights_only=True)
    assert all(
        torch.equal(value, latest["model"][key])
        for key, value in final["model"].items()
    )
    for name in ["config.json", "history.csv"]:
        shutil.copy2(ROOT / "results/pilot_seed0" / name, output / name)
    shutil.copy2(ROOT / "results/environment.json", output / "environment.json")
    card = """# Measured modular-addition pilot

These checkpoints belong to the 30,000-update pilot published at commit 35f56f8.
They are not checkpoints from the later reference-aligned experiment.

- Final training accuracy: 81.04%; validation accuracy: 63.02%.
- Grokking was not demonstrated; no held-out final test was evaluated.
- Configuration, history and original GPU environment accompany the weights.
- model_step_000000.pt and model_step_030000.pt contain configuration and model state.
- latest.pt also contains optimizer state and history for resumption.
- The repository verification records an exact CPU validation-accuracy match.
- SHA256SUMS covers every accompanying asset except the checksum file itself.

Download the three checkpoints, config.json and history.csv into runs/pilot_seed0,
then run `python -m grokking_lab analyze --name pilot_seed0 --all-checkpoints`.
Only initialization and final snapshots are available from this original backup.
Intermediate snapshots cannot be reconstructed from them.

Only load trusted PyTorch checkpoint files. See the repository report for the
split construction, interventions, limitations and scientific interpretation.
"""
    (output / "MODEL_CARD.md").write_text(card)
    checksums = "".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n"
        for p in sorted(output.iterdir())
    )
    (output / "SHA256SUMS").write_text(checksums)
    return sorted(output.iterdir())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backup", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    output = (
        args.output or Path(tempfile.mkdtemp(prefix="grokking-release-")) / "assets"
    )
    assets = prepare(args.backup, output)
    print("Verified release assets:", output)
    if args.publish:
        subprocess.run(["gh", "auth", "status"], check=True)
        subprocess.run(
            [
                "gh",
                "release",
                "create",
                TAG,
                *map(str, assets),
                "--repo",
                REPOSITORY,
                "--target",
                MEASURED_COMMIT,
                "--title",
                "Measured pilot checkpoints",
                "--notes-file",
                str(output / "MODEL_CARD.md"),
            ],
            check=True,
        )
        print(f"https://github.com/{REPOSITORY}/releases/tag/{TAG}")
    else:
        print(
            "Prepared only; no release was published. Add --publish after gh auth login."
        )


if __name__ == "__main__":
    main()
