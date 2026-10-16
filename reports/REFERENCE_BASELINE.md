# Reference-aligned baseline protocol

The next experiment targets a configuration documented in Neel Nanda's modular-addition walkthrough. It is separate from the measured 30,000-step pilot. A short CPU diagnostic now demonstrates near-perfect memorization, but the full baseline has not yet run and grokking is not claimed.

## Source and differences

Reference: [Grokking Demo at TransformerLens v1.6.0](https://github.com/TransformerLensOrg/TransformerLens/blob/54d548de4995a1ecc5b01b9c03aceaf0966c0eb3/demos/Grokking_Demo.ipynb), revision `54d548de4995a1ecc5b01b9c03aceaf0966c0eb3`. Its [configuration](https://github.com/TransformerLensOrg/TransformerLens/blob/54d548de4995a1ecc5b01b9c03aceaf0966c0eb3/transformer_lens/HookedTransformerConfig.py) sets the default initialization standard deviation to `0.8/sqrt(d_model)`, and its [initializer](https://github.com/TransformerLensOrg/TransformerLens/blob/54d548de4995a1ecc5b01b9c03aceaf0966c0eb3/transformer_lens/HookedTransformer.py) samples weight matrices from that normal distribution. The notebook freezes biases and computes log-softmax loss in float64.

| Setting | Original pilot | Published walkthrough | New baseline |
| --- | --- | --- | --- |
| Task and inputs | `(a+b) mod 113`, `[a,b,=]` | Same | Same |
| Layers, width, heads, MLP | 1, 128, 4 × 32, 512 | Same | Same |
| Activation and normalization | ReLU, no normalization | Same | Same |
| Matrix initialization std | 0.02 | `0.8/sqrt(128)` ≈ 0.07071 | 0.07071 |
| MLP biases | Trained | Frozen at zero | Frozen at zero |
| QKV/output/unembedding biases | Absent | Present but frozen at zero | Absent; equivalent zero contribution |
| Loss computation | Float32 | Float64 | Float64; model remains float32 |
| AdamW learning rate, decay, betas | 0.001, 1.0, `(0.9,0.98)` | Same | Same |
| Initialization seed | 0 | 999 | 999 |
| Data seed and partition | 2026; fixed validation/test, nested train pool | 598; random 30/70 split | Original project's 2026 partition preserved |
| Training horizon | 30,000 updates | 25,000 updates | 25,000 updates |
| Saved snapshot spacing | 1,000 updates | 100 updates | 1,000 updates plus initialization/final |
| Validation logging | Every 100 updates | Held-out loss every update | Every 100 updates |

This is **reference-aligned**, not a claim of an exact reproduction. The native PyTorch parameter layout and random-draw order differ from TransformerLens, so seed 999 does not give identical starting matrices. Most importantly, the original project's train/validation/test partition is preserved: a closer source split match would otherwise train on pairs previously reserved for final evaluation. Several settings change together as one baseline package; improvement cannot be attributed to initialization, biases or precision individually without follow-up ablations.

## Declared execution and analysis

- Configuration: `configs/reference.json`, initialization seed 999, full-batch training, fixed 25,000-update horizon. Output run: `reference_seed999` in a new directory.
- Validation and test identities are unchanged from the original pilot. Test accuracy remains uninspected.
- Report all logged training/validation loss and accuracy, parameter norm and timing. Use the final fixed-horizon model for the main result.
- A crossing is the first of three consecutive logged observations at or above 99%. Report train and validation crossings and their gap; a positive gap alone is not a complete circuit explanation. If validation does not cross within budget, report it as unobserved rather than extending silently.
- Save initialization, every 1,000 updates and final weights. Default checkpoint probes are at 0, 1,000, 5,000, 10,000 and 25,000 updates. Any later checkpoint selection based on observed curves is exploratory.
- Probe frequencies 1 through 5, ridge penalty 0.01, shuffled-label control seed 123. Fit on training pairs, score on validation pairs, export scores and settings. Do not choose a frequency using test performance.
- Keep the existing validation head ablations and embedding interventions, including FFT round-trip and five random-frequency controls. Random controls do not match energy.
- Leave the multi-seed sweep and final test disabled until the baseline is assessed. Future controlled comparisons must report every scheduled seed, including failures.

## Measured CPU diagnostic

Before committing a GPU budget, the new configuration was run to step 500 on CPU, with the horizon still recorded as 25,000. It reached 100% training accuracy at step 200 and sustained that through step 500; final validation accuracy was 1.84%. This demonstrates the memorization stage missing from the original pilot. It does **not** establish delayed generalization or guarantee that the full baseline will grok.

The diagnostic's settings, logs, environment and summary are in `results/reference_diagnostic`. It was stopped explicitly at step 500; it is not a completed 25,000-step experiment. No test evaluation was performed.

## Run

The portable notebook defaults to this new baseline in a fresh `grokking_lab_reference` directory on Colab/Kaggle. Select a GPU and Run all. It exports probes, plots and checkpoints together. The notebook remains concise by user preference.

CLI equivalent:

```bash
python -m grokking_lab train --config configs/reference.json --name reference_seed999
python -m grokking_lab analyze --name reference_seed999 --checkpoints 0 1000 5000 10000 25000
```

To checkpoint a partial diagnostic without changing the declared horizon, use `--stop-at 500`, then resume with the identical configuration and `--resume`. To extend a completed horizon, use the separate `extend` command and a new run name; it records the parent checkpoint checksum and the exploratory protocol change.
