# Colab pilot results

The user-supplied `grokking_experiments_export.zip` records one complete 30,000-update pilot and a 100-update smoke run. The pilot ended at 81.04% training accuracy and 63.02% validation accuracy. Neither reached the predeclared sustained 99% accuracy threshold. This experiment does not establish grokking or a recovered modular-addition circuit.

## Configuration and execution

The exported configuration uses modulus 113, width 128, four heads, MLP width 512, training fraction 0.30, split seed 2026, initialization seed 0, learning rate 0.001 and weight decay 1.0. The exported environment reports Python 3.13.15, PyTorch 2.11.0+cu130 and a Tesla T4. Recorded training elapsed time is 272.49 seconds; this is the training log's timing, not total notebook runtime.

All 301 expected observations from step 0 through step 30,000 are present. History values are finite, accuracies are bounded, timing is monotonic, and the final validation metrics agree with the intervention baseline. The FFT round-trip loss agrees within the notebook's tolerance. These checks establish internal consistency of the export, not independent reproduction of the GPU run. The source archive SHA-256 is recorded in `results/pilot_seed0/summary.json`. The original notebook source hash is `6483f29986b8edec41de6309d701811543bdab5e36a085ed5072b8d11ecfd94b`; the retained notebook is adapted from that supplied source.

| Measurement | Value |
| --- | ---: |
| Final training loss | 0.840807 |
| Final validation loss | 1.552093 |
| Final training accuracy | 81.04% |
| Final validation accuracy | 63.02% |
| Highest logged validation accuracy | 70.78% at step 9,400 |
| Highest logged training accuracy | 84.83% at step 9,400 |
| Sustained 99% training crossing | Not reached |
| Sustained 99% validation crossing | Not reached |

The highest logged checkpoint is descriptive only; the final fixed-horizon checkpoint remains the reported model. Accuracy fluctuates throughout training. There is no observed phase of near-perfect memorization followed by near-perfect delayed generalization under the specified criterion.

## Validation interventions

Removing individual heads reduces validation accuracy to 1.61%, 3.56%, 7.76% and 3.13%, respectively. Removing all heads yields 0.94%, compared with the baseline 63.02%. These interventions show sensitivity to each head in this model, without identifying a particular arithmetic circuit.

The five strongest nonzero embedding frequencies are 8, 16, 49, 56 and 24. Keeping these and the mean retains 82.77% of embedding energy, but achieves only 6.62% validation accuracy. Removing these frequencies gives 0.74%. Random five-frequency controls range from 0.63% to 8.38%; one exceeds the top-frequency-only condition. These observations do not establish that the selected frequency subset is sufficient or uniquely special. Energy is not matched across conditions, and embedding projection can move the model off its training distribution. Keep-top and remove-top both preserve the mean, so their retained energy fractions need not sum to one.

## Available artifacts and remaining work

The published pilot measurements and plots are under `results/pilot_seed0`; the GPU environment and package record are under `results`. Temporary smoke outputs and redundant local reports were archived outside the repository during cleanup.

The original results export contains no checkpoints or split-index files. A subsequent user-supplied `grokking_models_backup.zip` supplies initialization and step-30,000 model snapshots plus `latest.pt` with optimizer state. These model binaries are preserved outside the repository in the cleanup backup under Downloads. The original `grokking_models_backup.zip` also remains in Downloads. Restore its files alongside `results/pilot_seed0/config.json` and `history.csv` in a new run directory to analyze or resume; do not overwrite the published measurements. The final snapshot configuration matches the exported configuration, and its weights exactly match the latest checkpoint. An independent CPU validation evaluation reproduces 63.023893% accuracy exactly and loss 1.55209311 (within 0.000001 of the GPU result). The held-out test set was not accessed. See `checkpoint_verification.json` for backup provenance and checks. Split indices can be regenerated using the recorded configuration. Probe scores were not exported, so they are not reported here.

The export contains no multi-seed sweep and no final-test results. A single pilot does not support conclusions about seed variability or comparisons among weight-decay/training-size conditions. Further tuning should be labeled exploratory, keep the test set untouched, and record configuration changes. Code and these measured pilot findings can be published now with these limitations; the complete research protocol remains unfinished.

## Repository validation

After consolidation, all four automated tests pass, including execution of the portable notebook's smoke, validation-analysis and ZIP-export cells with a reduced CPU configuration, and verification of its GPU guard. Notebook schema, Python syntax, formatting, project metadata, JSON files and README/report links were checked. The published 30,000-step measurements were preserved; the long GPU experiment was not rerun during cleanup.
