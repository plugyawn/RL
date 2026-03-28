# Delightful GRPO GSM8K Screen (2026-03-28)

This project captures the first short-horizon Delightful GRPO screen we ran on top of NeMo-RL.

Artifacts from the screen live in [artifacts/](artifacts/), including:

- [metric_snapshot.json](artifacts/metric_snapshot.json)
- [metric_summary.json](artifacts/metric_summary.json)
- [stability_summary.json](stability_summary.json)
- [stability_summary.tsv](stability_summary.tsv)
- [validation__accuracy.png](artifacts/plots/validation__accuracy.png)
- [train__reward.png](artifacts/plots/train__reward.png)
- [train__dg_gate_mean.png](artifacts/plots/train__dg_gate_mean.png)
- [train__dg_gate_spread.png](artifacts/plots/train__dg_gate_spread.png)

## Setup

- Stack: this repo with the DG-GRPO changes in `nemo_rl/algorithms/loss/loss_functions.py`
- Model: `Qwen/Qwen2.5-Math-1.5B-Instruct`
- Task: built-in `gsm8k` train/test recipe
- Seed: `42`
- Horizon: `8` GRPO steps
- Batch shape: `4` prompts per step x `8` generations per prompt = `32` sampled responses per update
- Validation: `64` test samples every `4` steps
- Generation cap: `max_new_tokens=256`

This is a screen, not a claim-quality study. Across `8` steps we only touch about `32` unique GSM8K prompts.

## Outcome

| run | final val acc | best val acc | final reward | best reward |
|---|---:|---:|---:|---:|
| baseline | 0.34375 | 0.37500 | 0.50000 | 0.78125 |
| dg_eta1p0 | 0.31250 | 0.328125 | 0.50000 | 0.84375 |
| dg_eta0p5 | 0.37500 | 0.37500 | 0.40625 | 0.81250 |

Interpretation:

- `eta=1.0` underperformed the baseline on validation and is not the first DG setting to extend.
- `eta=0.5` is the only DG setting from this screen worth carrying into a longer run.
- The DG runs were stable, but the DG signal still looked weak.

## Rollout Stability

- All three runs completed without crashes, NaNs, or exploding losses.
- Final `train/gen_kl_error` stayed tiny:
  - baseline: `1.92e-4`
  - `dg_eta1p0`: `1.52e-4`
  - `dg_eta0p5`: `1.69e-4`
- `train/natural_termination_rate` stayed at `1.0` for all runs.
- Truncation stayed high at `max_new_tokens=256`:
  - baseline final `train/truncation_rate`: `0.71875`
  - `dg_eta1p0`: `0.78125`
  - `dg_eta0p5`: `0.68750`
- DG gate diagnostics were nearly neutral:
  - `dg_eta1p0` final `train/dg_gate_mean`: `0.49826`
  - `dg_eta0p5` final `train/dg_gate_mean`: `0.50047`
  - both DG runs had final `train/dg_gate_spread = 0.0`

The zero gate spread is the main caveat from this screen. Either the DG signal is genuinely weak in this tiny regime, or the masking/logging path still is not exposing the asymmetry we want.

## Reproduce

Baseline:

```bash
DG_ENABLED=0 RUN_NAME=gsm8k_baseline_seed42_screen MAX_STEPS=8 VAL_PERIOD=4 MAX_VAL_SAMPLES=64 \
  bash tools/run_dg_gsm8k_full.sh
```

DG (`eta=0.5`):

```bash
DG_ENABLED=1 DG_ETA=0.5 RUN_NAME=gsm8k_dg_eta0p5_seed42_screen MAX_STEPS=8 VAL_PERIOD=4 MAX_VAL_SAMPLES=64 \
  bash tools/run_dg_gsm8k_full.sh
```

Plot:

```bash
python tools/plot_grpo_ablation.py \
  --run baseline=/path/to/baseline \
  --run dg_eta1p0=/path/to/dg_eta1p0 \
  --run dg_eta0p5=/path/to/dg_eta0p5 \
  --metric validation/accuracy \
  --metric train/reward \
  --metric train/dg_gate_mean \
  --metric train/dg_gate_spread \
  --out-dir /tmp/dg_plots
```

Probe stability:

```bash
python tools/analyze_grpo_rollouts.py \
  --run baseline=/path/to/baseline \
  --run dg_eta1p0=/path/to/dg_eta1p0 \
  --run dg_eta0p5=/path/to/dg_eta0p5 \
  --out-dir /tmp/dg_analysis
```
