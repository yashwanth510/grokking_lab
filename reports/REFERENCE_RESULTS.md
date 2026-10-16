# Grokking observed in the reference-aligned run

The completed 25,000-update run with initialization seed 999 shows delayed generalization under the criterion declared in `REFERENCE_BASELINE.md`. Training accuracy first reaches sustained 99% at update 200; validation accuracy follows at update 8,800, an 8,600-update gap. Final training and validation accuracies are both 100%. This is one successful seed and condition, not a completed multi-seed study or an exact reproduction of the source paper.

![Reference-aligned learning curves](../results/reference_seed999/learning_curves.png)

## Measurement and verification

| Measurement | Recorded result |
| --- | ---: |
| Completed updates | 25,000 |
| Logged observations | 251 |
| Sustained 99% training crossing | 200 |
| Sustained 99% validation crossing | 8,800 |
| Crossing gap | 8,600 updates |
| Final training accuracy | 100% |
| Final validation accuracy | 100% |
| Final training loss | 9.2086 × 10⁻⁸ |
| Final validation loss | 0.000416528 |
| Recorded training time | 228.77 seconds on a Tesla T4 |

A crossing is the first observation of three consecutive logs at or above 99%. Updates are logged every 100 steps; the reported crossing is a logged boundary, not an exact continuous transition time. Training crossings at 200, 300 and 400 establish the first crossing; validation logs at 8,800, 8,900 and 9,000 establish the second. Validation remains above 99% at every subsequent logged observation.

The earlier history shows the separation clearly: validation accuracy is 1.76% at step 1,000, 6.54% at 5,000, 73.29% at 8,000 and 99.96% at 10,000, while training accuracy is already 100% at these checkpoints. The final model therefore succeeds on held-out pairs long after fitting the training pairs.

The user-supplied `grokking_lab_outputs.zip` contains all 26 expected snapshots from initialization through step 25,000, the optimizer checkpoint, split indices, environment, logs, figures and probe tables. The exported configuration exactly matches `configs/reference.json`. All logged values are finite, the step sequence is complete and timing is monotonic. The exported split arrays match regenerated arrays and the original pilot's train, validation and test identities exactly.

The final checkpoint loads with restricted PyTorch loading and its weights exactly match `latest.pt`. Independent CPU evaluation of the supplied final model reproduces 100% training and validation accuracy and the validation loss within 10⁻⁶. Additional CPU validation checks at initialization and steps 1,000, 5,000, 8,000, 9,000 and 10,000 are recorded in `checkpoint_validation_rechecks.csv`; per-checkpoint hashes and any numerical differences are included. This checks supplied weights and logs; training itself was not independently rerun.

Recorded time covers training/logging/checkpointing as measured by the notebook, not the entire notebook session. Export provenance and hashes are in `results/reference_seed999/summary.json`; verification software and checks are in `checkpoint_verification.json`. **No final held-out test evaluation was performed.**

## Interpretability results

Removing heads 0, 1, 2 and 3 yields validation accuracies of 41.21%, 96.40%, 36.62% and 36.90%, respectively. Removing all heads yields 0.86%. The model is sensitive to attention interventions; the smaller effect of removing head 1 suggests a smaller individual contribution under this specific ablation, not proof that the head is unnecessary in every context.

The five largest nonzero embedding frequencies are 38, 9, 25, 52 and 43. Keeping these frequencies plus the mean retains 96.47% of embedding energy and preserves 100% validation accuracy. Removing them reduces validation accuracy to 0.90%. Five random same-count frequency controls achieve approximately 0.74–0.98% accuracy.

This is stronger evidence for a useful selected embedding-frequency subspace than in the unsuccessful pilot: keeping it preserves predictions and removing it disrupts them. It is not full recovery of the network's arithmetic algorithm. The top-frequency condition retains far more energy than random controls; energy is not matched, the downstream network is unchanged, and weight projection can move representations off distribution. Further controls and a trace through attention, MLP and output weights are needed for a circuit claim.

The predeclared ridge probes use frequencies 1–5, penalty 0.01, training-only fitting and validation scoring, with shuffled-target seed 123. Frequency 1 validation R² changes from −0.0218 at initialization to 0.7874 at step 10,000 and 0.7264 at step 25,000; the final shuffled-control score is −0.1076. These probes use different frequencies from the selected embedding intervention and indicate linear accessibility rather than proving use in prediction. Probe tables at all five declared checkpoints are exported under `analysis/`.

## Artifacts and limits

Small measured artifacts are in `results/reference_seed999`. The full original output archive, including all snapshots and optimizer state, is distributed through the [reference-v1 release](https://github.com/yashwanth510/grokking_lab/releases/tag/reference-v1), with checksums and a model card. Model binaries remain outside ordinary source commits.

The [original unsuccessful pilot](RESULTS.md) is retained. Several settings changed together for this reference-aligned configuration, including initialization scale, bias training, loss precision and initialization seed. The comparison does not isolate which change caused the different outcome. The preserved split and the source-configuration deviations are documented in `REFERENCE_BASELINE.md`.

The full project still requires repeated seeds and controlled data-size/weight-decay comparisons, including failures. The final test remains untouched. A successful single-seed validation result does not establish reliability across seeds, novelty, a recovered Fourier circuit or completion of the planned research study.
