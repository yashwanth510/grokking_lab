# Grokking Research Lab

Modular-addition experiments with a small causal Transformer: training curves, Fourier analysis, probes and interventions. This is an independent, paper-inspired implementation.

The completed reference-aligned run **shows grokking under the declared criterion**: sustained 99% training accuracy at step **200**, sustained 99% validation accuracy at **8,800**, an **8,600-update delay**, and **100% final training and validation accuracy** after 25,000 updates. The original pilot remains documented at 81.04% training and 63.02% validation accuracy. This is one successful seed; multi-seed comparisons and final test evaluation remain unrun.

![Reference-aligned learning curves](results/reference_seed999/learning_curves.png)

Read [the grokking results](reports/REFERENCE_RESULTS.md), [the original pilot report](reports/RESULTS.md), and [the baseline protocol and reference differences](reports/REFERENCE_BASELINE.md).

## Run the new baseline

Use the single portable [Grokking_Research_Lab.ipynb](notebooks/Grokking_Research_Lab.ipynb).

- **Colab:** upload it at [Google Colab](https://colab.research.google.com/), select **Runtime → Change runtime type → GPU**, then **Run all**.
- **Kaggle:** import the same file, select a GPU accelerator, then run all cells.

The notebook generates its data and needs no credentials or external dataset. It uses installed PyTorch, NumPy, pandas and matplotlib. Defaults are `RUN_PILOT = True`, `RUN_SWEEP = False` and `FINAL_PROTOCOL_FROZEN = False`.

It runs a 100-step smoke check, the **25,000-step reference-aligned baseline**, validation analyses and probes at several saved checkpoints, then exports results and model weights together. It saves probe frequencies, ridge penalty, ordinary and shuffled scores to CSV. It also saves the pilot learning curve. For a CPU smoke check, set `RUN_PILOT = False`.

New notebook outputs go to `/content/grokking_lab_reference` in Colab, `/kaggle/working/grokking_lab_reference` in Kaggle, or `runs/reference` locally. These names keep the original pilot separate. The final cell downloads a ZIP in Colab or displays a file link elsewhere. Save the ZIP before ending a temporary runtime. To resume, restore `grokking_lab_reference` under `/content` or `/kaggle/working` and run again with unchanged settings; existing checkpoints resume automatically.

This configuration follows the reference's initialization scale, frozen biases and loss precision, while preserving the original project's held-out test pairs. It is **not an exact reproduction**, and several changes are combined; individual causes of any improvement require controlled ablations. See the protocol for all differences.

## Run locally

Use Python 3.10 or later and a PyTorch installation suitable for your hardware.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q

python -m grokking_lab train --config configs/smoke.json --name smoke_seed0
python -m grokking_lab analyze --name smoke_seed0
```

CLI runs go under `runs/`, separate from published measurements in `results/`.

```bash
python -m grokking_lab train --config configs/reference.json --name reference_seed999
python -m grokking_lab train --config configs/reference.json --name reference_seed999 --resume
python -m grokking_lab analyze --name reference_seed999 --checkpoints 0 1000 5000 10000 25000
```

Analysis writes `analysis/step_XXXXXX/` tables and plots plus `analysis/checkpoint_summary.csv`. Existing published summaries and original GPU exports are preserved. `--all-checkpoints` analyzes every available snapshot. Read checkpoints from a separate backup with `--checkpoint-dir PATH`.

For a controlled partial run, add `--stop-at 500`; resume with the same configuration to continue to the declared horizon. To extend an existing horizon, use a new directory:

```bash
python -m grokking_lab extend --source runs/reference_seed999 --name reference_extended --steps 50000
```

Extensions preserve the parent run and record its checkpoint checksum and the exploratory protocol change. They are not evidence from the original fixed-horizon experiment. The original experiment remains reproducible with `configs/pilot.json`.

## Download the measured models

The successful run is available in the [reference-v1 release](https://github.com/yashwanth510/grokking_lab/releases/tag/reference-v1): the complete output archive includes 26 snapshots, optimizer state, split arrays, logs, environment, plots and probe tables. Download the archive and verify it against `SHA256SUMS`. Extract it to restore `grokking_lab_reference` in Colab or Kaggle; no further training is needed to inspect the final model.


The original pilot's initialization snapshot, final snapshot and resumable checkpoint are public in the [pilot-v1 release](https://github.com/yashwanth510/grokking_lab/releases/tag/pilot-v1), alongside configuration, training history, GPU environment, model card and SHA-256 checksums. These are the original pilot's weights, not the new baseline's weights.

With GitHub CLI installed:

```bash
mkdir -p runs/pilot_seed0
gh release download pilot-v1 --repo yashwanth510/grokking_lab --dir runs/pilot_seed0
(cd runs/pilot_seed0 && sha256sum -c SHA256SUMS)
python -m grokking_lab analyze --name pilot_seed0 --all-checkpoints
```

Load only trusted checkpoints. Model binaries remain outside ordinary source commits. The original backup includes only initialization and final snapshots, so intermediate states from that run cannot be recovered.

## Method and evaluation

Inputs are `[a,b,=]`; labels are `(a+b) mod 113`. Split seed 2026 reserves 20% each for validation and test, with nested training subsets from the remaining pool. Both pilots use 30% of all pairs. Reversed pairs can cross splits. Initialization and split seeds are separate.

The model has one causal attention block and an MLP, width 128, four heads, MLP width 512, and no LayerNorm or dropout. AdamW uses learning rate 0.001, betas `(0.9,0.98)` and decay 1.0. Training is full-batch. The original pilot uses initialization std 0.02, trainable MLP biases and float32 loss. The new configuration uses std `0.8/sqrt(128)`, frozen zero biases and float64 loss; model parameters remain float32.

A crossing is the first logged step at or above 99% accuracy for three consecutive observations. Missing crossings mean the threshold was not sustained within budget. Fourier peaks and probes do not identify a causal circuit. Projection interventions can move the model off distribution; random-frequency controls do not match retained energy.

Keep the final test untouched while developing the baseline. After freezing experiment and analysis choices, explicitly evaluate the final checkpoint:

```bash
python -m grokking_lab test --name reference_seed999 --protocol-frozen
```

This refuses to overwrite an existing `final_test.json`. The optional notebook sweep covers five fraction/decay conditions and three seeds; it remains disabled until the baseline is assessed. Equal updates do not mean equal compute.

## Repository

```text
configs/           Original smoke/pilot and new baseline settings
notebooks/         One portable, concise experiment notebook
src/grokking_lab/  Training and analysis package
tests/             Data, model, resume, analysis and notebook checks
scripts/           Notebook synchronization and model-release preparation
results/           Original pilot, successful GPU baseline and CPU diagnostic
reports/           Findings and reference-baseline protocol
.github/workflows/ GitHub checks
```

The notebook is self-contained for upload. Its shared definitions are generated from the package; after editing the package, run `python scripts/sync_notebook.py`. CI checks synchronization. Temporary runs, caches and model binaries are ignored. `results/pip_freeze.txt` records the original GPU environment, not a portable dependency lock. Code is provided under the [MIT license](LICENSE).

## References

- Nanda et al., [Progress measures for grokking via mechanistic interpretability](https://arxiv.org/abs/2301.05217).
- Neel Nanda, [Modular addition walkthrough](https://www.neelnanda.io/mechanistic-interpretability/modular-addition-walkthrough).
- [Pinned TransformerLens v1.6.0 training notebook](https://github.com/TransformerLensOrg/TransformerLens/blob/54d548de4995a1ecc5b01b9c03aceaf0966c0eb3/demos/Grokking_Demo.ipynb).
