# Core Research Playground

This directory decouples the **toric model core** from the trading pipeline and
answers one question honestly: *does the invented architecture earn its
complexity anywhere?* Everything here is reproducible from a fresh clone with
CPU-only PyTorch, in minutes.

## Contents

| File | Purpose |
|------|---------|
| `toric_core.py` | Cleaned-up toric encoder: angle-parameterized rotations, proper Markov kernel option, geodesic gating, optional amplitude bypass, hybrid forecaster |
| `baselines.py` | GRU / LSTM / Transformer / DLinear / persistence / drift / seasonal-naive |
| `train_utils.py` | Shared windowing, train-only standardization, AdamW + early stopping |
| `synthetic_torus.py` | Home-turf task: Markov-switching rotations on a torus |
| `ett_experiment.py` | Real-data pilot: ETTh1, canonical 96->96 protocol |

## What was cleaned up vs the legacy core

1. **Rotations are angles now.** Legacy `FractalToricCell` used unconstrained
   complex parameters (`randn * 0.1`) whose magnitudes were discarded by
   renormalization — dead degrees of freedom. Now `theta[state, dim]` is an
   angle and the rotation is `exp(i*theta)`: unit modulus by construction.
2. **The chain can be an actual Markov chain.** `kernel="stochastic"` composes
   a row-stochastic transition matrix with observation evidence
   (`p' ~ (p @ P) * exp(logits)`, an HMM-style filter); the legacy unconstrained
   `p @ A` remains as `kernel="free"`. `transition_matrix()` exposes P.
3. **Gating is a torus geodesic** (convex mix of unit vectors renormalized —
   exact slerp now that both operands are unit modulus by construction).
4. **Optional amplitude bypass** — the pure phase pipeline discards magnitudes
   entirely; cyclic data may not care, amplitude-heavy data does.
5. 8-bit straight-through angle quantization kept, but switchable
   (`--no-quant` uses a 16-bit grid) so its effect is measurable.

## Results

### 1. Home turf: Markov-switching rotation on a torus

Data: 8 dims, 4 sticky regimes (P=0.9), regime-specific angular velocities,
`x_t = amp * sin(theta_t) + 0.1 eps`. Predicting the horizon requires tracking
phases AND the regime — exactly the model's inductive bias.

Easy task (T=64, H=12, 6 epochs, 2,500 train sequences):

| model | test MSE | params |
|-------|----------|--------|
| **DLinear (linear AR)** | **0.1112** | 1,560 |
| persistence | 0.1272 | 0 |
| GRU | 0.4915 | 14,988 |
| LSTM | 0.4917 | 19,724 |
| Transformer | 0.4925 | 101,068 |
| MMT (toric) | 0.5187 | 39,782 |
| drift | 1.3532 | 0 |

Hard task (T=96, H=24, speeds x2, 25 epochs):

| model | test MSE | params |
|-------|----------|--------|
| **DLinear** | **0.3388** | 4,656 |
| hybrid (linear base + toric correction) | 0.3743 | 46,466 |
| GRU | 0.5043 | 15,768 |
| Transformer | 0.5070 | 101,848 |
| MMT (toric) | 0.5324 | 40,946 |
| persistence | 0.5574 | 0 |

### 2. Optimization probe: can the recurrent models even fit 300 sequences?

50 epochs, lr 3e-3, no early stopping (train MSE is the diagnostic):

| model | train MSE | test MSE |
|-------|-----------|----------|
| DLinear | **0.101** | 0.098 |
| GRU | 0.484 | 0.493 |
| MMT | 0.510 | 0.526 |
| MMT (no quantization, free kernel, amplitude path) | 0.489 | 0.497 |

Neither the GRU nor the toric model can overfit 300 sequences; both plateau at
"predict the unconditional mean". Removing quantization does not unblock
learning, so the blocker is not the STE — gradient descent on phase-rotation
dynamics is the difficulty. The linear AR solves the same task to 0.10 because
a fixed-frequency sinusoid satisfies a 2-term linear recurrence.

### 3. Real data: ETTh1, canonical 96 -> 96 protocol

Split 12/4/4 months, z-score by train stats, MSE on standardized scale.
Published reference for DLinear on this protocol: ~0.375-0.41 — our harness
reproduces it, which validates the setup.

| model | test MSE | test MAE | params |
|-------|----------|----------|--------|
| **DLinear** | **0.3904** | 0.4048 | 18,624 |
| seasonal-naive (24h) | 0.5122 | 0.4333 | 0 |
| hybrid | 0.7787 | 0.6267 | 87,906 |
| GRU | 0.7821 | 0.6731 | 57,696 |
| MMT (toric) | 1.0873 | 0.7919 | 69,282 |

## Verdict (Round 1)

1. On quasi-periodic data — synthetic *and* real — a per-channel linear
   autoregression (DLinear) is the strongest model at every tested budget.
   This mirrors the published DLinear result and extends it to the toric
   architecture: **the toric model currently adds no measurable value over a
   linear baseline**, on its own home turf or on ETTh1.
2. The zero-initialized toric correction in the hybrid does not find signal
   beyond what the linear base already captures.
3. The failure is an *optimization* failure shared with GRU/LSTM/Transformer,
   not a trading problem and not a data-volume problem. Switching application
   domain will not fix it by itself.

## Round 2: the three roadmap fixes, tested (diff inputs, residual training, race)

All three planned fixes were implemented and measured. **None of them makes
the toric core competitive; they do clarify where the problem lives.**

Implemented:

- `use_diff_input` on ToricEncoder / GRU / LSTM / Transformer: first
  differences are appended to the inputs, exposing the local phase increment
  (the regime identifier) directly to the model.
- `train_hybrid_residual` (``hybrid_res``): phase 1 trains the linear base
  alone (the frozen toric path is zero-initialized, i.e. exactly a DLinear);
  phase 2 freezes the base and trains the toric encoder + correction head on
  the base's residual `y - base(x)`.  The correction mathematically cannot
  destabilize the base.
- Curriculum for MMT: warm-up on half-speed rotations, then the real task.
- Convergence race: 30 epochs, patience 5, seeds 0 and 1, diff+amplitude on
  for all neural models.

### Synthetic home turf (T=64, H=12, 2,500 sequences)

| model | seed 0 test MSE | seed 1 test MSE |
|-------|-----------------|-----------------|
| DLinear | **0.0992** | **0.0988** |
| hybrid_res | 0.1021 | 0.1016 |
| GRU (+diff) | 0.4906 | 0.4905 |
| MMT (+diff) | 0.4935 | 0.4941 |
| MMT (+diff, curriculum) | 0.4935 | — |

Findings:

1. **Diff inputs do not unblock the recurrent models.** GRU/MMT stay at the
   ~0.49 "predict the mean" plateau.  The information needed (phase and speed
   are directly recoverable from `(x, dx)`) is present in the inputs; gradient
   descent still does not find the solution.
2. **Curriculum does not unblock it either** (0.4935, unchanged).
3. **The two-phase residual mechanism works exactly as designed** — the base
   matches standalone DLinear and the correction cannot hurt it — **but the
   toric core extracts almost nothing from the residual**: residual val MSE
   moves 0.1085 -> 0.1080 (R^2 ~ 0.5%).  On this task the linear base sits at
   the achievable floor; there is no structured residue for the torus to
   explain.

### ETTh1 96 -> 96 (same protocol as Round 1)

| model | test MSE | test MAE |
|-------|----------|----------|
| DLinear (round 1) | **0.3904** | 0.4048 |
| hybrid_res | 0.4031 | 0.4131 |
| GRU (+diff) | 0.8012 | 0.6906 |
| MMT (+diff) | 1.0242 | 0.7550 |

The residual-based hybrid lands within noise of pure DLinear (the correction
costs ~0.013 MSE here).  Note the diagnostic printed before phase 2: the
linear base leaves a residual val MSE of ~0.67 on ETTh1 — **real structure
remains after the linear model — but the toric core cannot extract it.**
A stronger residual learner (e.g. gradient boosting over lag/cycle features,
as already used elsewhere in this repository) is the natural next experiment.

## Verdict (after Round 2)

1. On quasi-periodic data — synthetic *and* real — a per-channel linear
   autoregression (DLinear) remains the strongest model at every tested
   budget, and on the home-turf synthetic task it sits at the achievable
   floor.  This mirrors the published DLinear result.
2. The toric model's failure is an *optimization* failure shared with
   GRU/LSTM/Transformer, not specific to the torus and not a data-volume or
   domain problem: diff inputs and curriculum do not unblock learning, and
   neither the toric core nor generic RNNs can overfit 300 sequences.
3. The residual two-phase hybrid is the right *safety* architecture (the
   toric path can only help or do nothing), but the toric core currently
   explains ~0.5% of the residual variance on synthetic data and slightly
   hurts on ETTh1.

## What to try next (revised after Round 2)

1. **Design a task where linear AR provably fails** — regime-conditional
   nonlinear structure (e.g. regime-dependent amplitude or frequency chirps
   with identifiable regimes) — and test whether the toric core, which is the
   only model here with an explicit discrete-state mechanism, wins there.
   This is now the *only* path to justifying the architecture.
2. **Try a stronger residual learner on the ETTh1 residual** (val MSE ~0.67
   after the linear base): gradient boosting over lag/cycle features — the
   repository already uses LightGBM elsewhere — to check whether the residual
   is machine-learnable at all before blaming the toric core.
3. **Convergence race with a real budget** (GPU, 100+ epochs, lr sweep,
   several seeds) before any claim about ceilings; the CPU race here is
   consistent across two seeds but small.
4. Only after the toric core beats DLinear on at least one task a mixture of
   linear models cannot represent, move to a real domain: seasonal load
   forecasting, ECG/HRV phase analysis, or anomaly detection in periodic
   machinery signals.

## Reproduce

```bash
# Round 1
python -m research.synthetic_torus --models mmt,gru,dlinear,naive --epochs 6
python -m research.synthetic_torus --models mmt,hybrid,dlinear,naive --epochs 25 --patience 5 \
    --speed-scale 2.0 --horizon 24 --seq 96 --out research/results_synthetic_hard.json
python -m research.synthetic_torus --models mmt,gru,dlinear --n-train 300 --n-val 100 --n-test 100 \
    --epochs 50 --patience 50 --lr 3e-3 --out research/results_overfit_probe.json
python -m research.ett_experiment --models dlinear,gru,seasonal --epochs 8
python -m research.ett_experiment --models mmt,hybrid --epochs 6

# Round 2 (diff inputs, residual hybrid, curriculum, race)
python -m research.synthetic_torus --models dlinear,gru,mmt --epochs 30 --patience 5 \
    --diff-input --amplitude --seed 0 --out research/results_race_s0_a.json
python -m research.synthetic_torus --models hybrid_res --epochs 30 --patience 5 --base-epochs 10 \
    --diff-input --amplitude --seed 0 --out research/results_race_s0_b.json
python -m research.synthetic_torus --models dlinear,gru,mmt,hybrid_res --epochs 30 --patience 5 \
    --base-epochs 10 --diff-input --amplitude --seed 1 --out research/results_race_s1.json
python -m research.synthetic_torus --models mmt --epochs 30 --patience 5 --curriculum \
    --diff-input --amplitude --seed 0 --out research/results_curriculum.json
python -m research.ett_experiment --models hybrid_res,mmt,gru --epochs 6 --base-epochs 4 \
    --diff-input --out research/results_ett_round2.json
```

Result JSON files are committed next to this README.
