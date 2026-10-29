# Core Research Playground

This directory decouples the **toric model core** from the trading pipeline and
answers one question honestly: *does the invented architecture earn its
complexity anywhere?* Everything here is reproducible from a fresh clone with
CPU-only PyTorch, in minutes.

## Contents

| File | Purpose |
|------|---------|
| `toric_core.py` | Cleaned-up toric encoder: angle-parameterized rotations, proper Markov kernel option, geodesic gating, optional amplitude bypass, hybrid forecaster |
| `baselines.py` | GRU / LSTM / Transformer / DLinear / MLP / mixture-of-linear-experts / persistence / drift / seasonal-naive |
| `train_utils.py` | Shared windowing, train-only standardization, AdamW + early stopping, two-phase residual hybrid training |
| `synthetic_torus.py` | Home-turf task: Markov-switching rotations on a torus |
| `nonlinear_torus.py` | Roadmap item 1: regime-conditional nonlinear task (regime-dependent amplitude + rotation rate) where a single linear map is mis-specified |
| `ett_experiment.py` | Real-data pilot: ETTh1, canonical 96->96 protocol |
| `ett_residual_lgbm.py` | Roadmap item 2: LightGBM over lag/envelope/calendar features on the post-linear ETTh1 residual — is the residual machine-learnable at all? |
| `overfit_budget.py` | Roadmap item 3: long-budget overfit probe (lr sweep x seeds x 200 epochs) on the 300-sequence task |
| `probe_state.py` | Round 3.5 control 1: linear probes (regime / phase / forecast readout) on the frozen MMT state |
| `phase_rnn.py` | Round 3.5 control 2: minimal fully-differentiable rotation recurrence — overfit probe + regime task |

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

> **Harness correction (Round 3.5).** Rounds 1–3 contained a harness bug:
> on the synthetic tasks the forecast heads of MMT / GRU / LSTM /
> Transformer / hybrid were built with the default `num_channels=1`, so the
> model emitted a single `(B, H, 1)` series that `mse_loss` broadcast across
> all channels.  Such a model can only predict the cross-channel mean
> sequence; its best achievable loss is the cross-channel-mean floor (home-
> turf easy task: ~0.47 — the observed "~0.50 plateau" was that floor, not a
> gradient barrier).  `ett_experiment` was unaffected (it passes
> `num_channels=7` explicitly), so **all ETTh1 numbers and all DLinear / MLP /
> MoE-linear / persistence / drift / analytic-readout numbers below stand**;
> every synthetic MMT / GRU / hybrid number in Rounds 1–3 is superseded by
> the corrected Round 3.5 runs at the bottom of this file.

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

## Round 3: the three roadmap items, tested (nonlinear task, residual learnability, long budget)

All three items from the revised roadmap were implemented and measured on the
same CPU-only setup (no GPU was available; item 3 was therefore sized so that
one run stays minutes-long — the harness is budget-generic).

### Item 1 — a task where linear AR fails by design (`nonlinear_torus.py`)

Generative process: a hidden regime s_t follows a sticky Markov chain
(K=3, self-transition 0.97, mean sojourn ~33 steps); each regime has its own
amplitude a_k in {0.6, 1.4, 2.4} and angular velocity w_k in
{0.05, 0.20, 0.45} rad/step; the observation is the full complex readout
x_t = a_{s_t} * (cos theta_t, sin theta_t) + noise (sigma=0.05).  The optimal
predictor is a regime-posterior mixture of rotated/scaled copies of the last
point — no single linear operator on the window can represent it (regime
evidence is envelope/rotation-rate structure, i.e. quadratic and angular in
the samples).  Windows are mostly single-regime, so "identify the regime,
then apply its dynamics" is well defined.

Task: T=64, H=12, 2,500 train sequences, 30 epochs, patience 5, seed 0.
The no-learning *phase readout* estimates phase/envelope/rate from the last
8 steps and propagates — a structure-aware reference proving the information
is in the window (it uses generator-matched knowledge the learned models are
not given).

| model | test MSE | params |
|-------|----------|--------|
| MLP (flattened window) | **0.3725** | 36,120 |
| MoE-linear (linear experts + nonlinear gate) | 0.3957 | 17,739 |
| phase readout (no learning) | 0.5311 | 0 |
| DLinear | 0.6123 | 1,560 |
| GRU (+diff) | 0.8737 | 14,220 |
| MMT (toric, +diff, +amp) | 0.8963 | 39,878 |
| MMT (exactly 3 chain states) | 0.9236 | 36,266 |
| persistence | 2.8079 | 0 |

Findings:

1. **The task separates linear from nonlinear exactly as designed**: DLinear
   pays +64% MSE vs the MLP — the regime-conditional structure is real,
   identifiable and extractable.
2. **The minimal regime mechanism is sufficient**: MoE-linear (three linear
   experts behind a two-layer MLP gate) matches the plain MLP.  Nothing more
   expressive than "detect regime, then linear" is needed.
3. **The toric model loses on a task built from its own assumptions.**  It
   trails the MLP by a factor of 2.4 and even the linear baseline by 46%.
   With exactly as many chain states as true regimes, the unsupervised chain
   recovers them at 0.502 agreement (chance = 0.33; provenance: this is the
   model's own argmax over its chain posterior — no trained probe involved)
   — above chance, far from
   usable.  The no-learning phase readout beats every gradient-trained
   recurrent model: the failure is gradient descent on the phase-recurrent
   parameterization, not the information content of the task.

### Item 2 — is the post-linear ETTh1 residual machine-learnable at all? (`ett_residual_lgbm.py`)

The DLinear base was retrained with the identical protocol (it reproduces the
round-1 test MSE 0.3904 exactly, validating the harness) and frozen.  One
LightGBM model per channel (250 trees max, early stopping, lr 0.05) predicts
the base's residual from 107 features per (window, step): the 96 raw lags,
6 envelope statistics (mean/std/last/last-diff/RMS-24/mean-abs-diff), the
forecast step index, and the target timestamp's hour-of-day and day-of-week
phase (sin/cos).  Training rows use every 3rd window (overlapping windows are
highly redundant); early stopping on a held-out val subsample.

| quantity | value |
|----------|-------|
| residual val MSE after base | 0.6566 |
| residual val MSE after +LightGBM | 0.6404 |
| **residual R^2 (val, overall)** | **0.025** |
| base test MSE / MAE | 0.3904 / 0.4048 |
| base+LightGBM test MSE / MAE | 0.3843 / 0.4054 |

Per-channel residual R^2: HUFL 0.059, LULL 0.058, MUFL 0.053, OT 0.012,
MULL 0.004, HULL -0.005, LUFL -0.006.  Early stopping keeps only 1-18 trees
per channel; the largest gains come from window statistics (w_std, w_mad,
w_mean), not calendar phase.

**Finding: ~97.5% of the post-linear residual variance is unexplainable by
lag/envelope/calendar features for a strong tabular learner.**  Round 2's
accusation against the toric core is defused: there is essentially no
structure left for ANY model in this feature family — the per-channel linear
base is the practical ceiling on this benchmark.  (The residual caveat: the
val-based early stopping is conservative, so the small positive R^2 on some
channels may be slightly underestimated, but not by an order of magnitude.)

### Item 3 — the overfit probe at a real budget (`overfit_budget.py`)

Round 1's optimization probe (50 epochs, single lr, single seed) is extended
to 100 epochs, a learning-rate sweep {1e-3, 3e-3, 1e-2} and seeds {0, 1},
with early stopping disabled (patience = epochs) so the budget is actually
spent.  Same 300-sequence task; train MSE is the diagnostic.

| model | best config | min train MSE | test MSE |
|-------|-------------|---------------|----------|
| DLinear | lr 3e-3, seed 1 | 0.0959 | 0.0983 |
| MLP | lr 3e-3, seed 0 | **0.0290** | 0.1536 |
| GRU | lr 1e-3, seed 1 | 0.4823 | 0.4914 |
| MMT (toric) | lr 3e-3 | 0.4963 - 0.5009 | 0.5131 - 0.5181 |

(MMT at lr 1e-2: min train 0.5109 / 0.5303 test — worse, not better.  The
mean-predictor reference on this split: val MSE 0.5679.)

Findings:

1. **The plateau is not a budget artifact.**  100 epochs x 3 learning rates x
   2 seeds do not move the toric model (or the GRU) off the ~0.49-0.51
   "predict the mean" plateau on 300 sequences.
2. **It is not "small data defeats nonlinear models" either.**  The flattened
   MLP overfits the same 300 sequences to 0.029 train MSE — 16x below the
   plateau — and still generalizes (test 0.154).  DLinear fits as always.
3. **The barrier is specific to the recurrent parameterization.**  Gradient
   descent through phase-rotation dynamics (toric) or gated linear recurrences
   (GRU) cannot even memorize 300 sequences at any tested configuration,
   while a flat nonlinear map memorizes them trivially.

## Verdict (after Round 3)

1. **Roadmap item 1 produced the intended experiment and the intended
   separation — but not the intended winner.**  On a regime-conditional task
   where linear AR provably pays +64%, the toric core loses to every learned
   baseline, including the minimal regime mechanism (MoE-linear) built from
   the same "discrete latent state" idea.  The architecture's claimed
   inductive bias is exactly what the task rewards, and training still cannot
   exploit it.
2. **Roadmap item 2 closed the real-data question.**  The post-linear ETTh1
   residual is ~97.5% unlearnable by gradient boosting over lag/envelope/
   calendar features.  There is no hidden structure for the toric core (or
   anything else) to extract on this benchmark.
3. **Roadmap item 3 localized the failure.**  The toric model shares the
   GRU's optimization pathology — unable to overfit 300 sequences under any
   tested budget — while a flat MLP overfits them 16x below the plateau.
   The bottleneck is gradient flow through the phase-recurrent
   parameterization, not data volume, not quantization alone, not task choice.
4. **Combined verdict: across home-turf synthetic data, its most favorable
   regime-conditional task, and real seasonal data, the toric architecture
   adds parameters and training difficulty without a single measurable win.**
   On the evidence collected in three rounds, it should not be used as a
   forecaster.

## Round 3.5: harness correction, re-runs, and the two pre-Round-4 controls

Three things happened between Round 3 and Round 4: the residual experiment
was hardened with the controls a reviewer asks for (seeds, shuffle, direct
fit), a **harness bug was found that invalidates the Round 1–3 optimization
story**, and the two cheap pre-Round-4 controls were run.  All corrected
result JSONs are committed next to this README (`*_fixed.json`,
`results_probe_*.json`, `results_phase_rnn_*.json`, and the hardened
`results_ett_residual_lgbm.json`).

### The broadcast-head bug

On the synthetic harnesses the heads of MMT / GRU / LSTM / Transformer /
hybrid were constructed with the default `num_channels=1`.  Their output
`(B, H, 1)` was silently broadcast across channels by `mse_loss`, so those
models were trained to predict a *single shared series* fitted to the
cross-channel mean of the targets.  Consequences, now measured:

- The home-turf easy-task broadcast floor is ~0.47 — the "0.49–0.51
  predict-the-mean plateau" of MMT and GRU in Rounds 1–3 was exactly that
  floor.  It said nothing about gradient flow through phase dynamics.
- `ett_experiment` already passed `num_channels=7`, so the ETTh1 table was
  never affected.  DLinear (channel-shared map, correct output shape),
  MLP and MoE-linear (constructed with explicit `num_channels`), the naive
  predictors and the analytic readout were never affected.
- The 0.502 regime-agreement figure is unaffected (it is produced by the
  model itself — argmax of its chain posterior — and the posterior does not
  pass through the head at all) — but see Control 1 below for what the
  probes now say about it.

All synthetic heads now emit `(B, H, C)` with `C` = number of channels.

### Corrected headline tables

Nonlinear regime task (`nonlinear_torus.py`, T=64, H=12, 2,500 sequences,
30 epochs, seed 0 — fixed heads everywhere):

| model | old (buggy head) | **fixed head** | params |
|-------|-----------------|----------------|--------|
| GRU (+diff) | 0.8737 | **0.3567** | 15,000 |
| MLP (flattened window) | 0.3725 (valid) | 0.3725 | 36,120 |
| MMT (toric, +diff, +amp) | 0.8963 | **0.3878** | 41,042 |
| MoE-linear | 0.3957 (valid) | 0.3957 | 17,739 |
| MMT (exactly 3 chain states) | 0.9236 | **0.4363** | 37,430 |
| phase readout (no learning) | 0.5311 (valid) | 0.5311 | 0 |
| phase-RNN, linear kick (Round-4 candidate) | — | 0.6386 | 12,408 |
| DLinear | 0.6123 (valid) | 0.6123 | 1,560 |
| phase-RNN, mlp kick (Round-4 candidate) | — | 0.6932 | 15,608 |
| persistence | 2.8079 (valid) | 2.8079 | 0 |

The Round 3 headline — "the toric model loses on a task built from its own
assumptions, to every learned baseline" — was an artifact.  With a head that
can address channels, the toric model lands within ~4% of the MLP and ~9% of
the GRU, beats the minimal regime mechanism (MoE-linear) and beats the
analytic readout by 27%.  The GRU-vs-MMT ordering (0.357 vs 0.388) is the
honest remaining gap.

Home-turf task (`synthetic_torus.py`, fixed heads; DLinear/persistence rows
unchanged):

| task | model | old | **fixed** |
|------|-------|-----|-----------|
| easy (T=64, H=12) | DLinear | 0.1112 | 0.1112 |
| | GRU | 0.4915 | **0.1170** |
| | persistence | 0.1272 | 0.1272 |
| | MMT | 0.5187 | **0.3852** |
| hard (T=96, H=24, speeds x2) | DLinear | 0.3388 | 0.3388 |
| | GRU | 0.5043 | **0.2622** |
| | MMT | 0.5324 | **0.4815** |
| | persistence | 0.5574 | 0.5574 |

The toric model still clearly loses the 8-channel home-turf tasks to GRU and
DLinear — that gap survives the head fix and is an optimization gap (see
next table), not an output-cap artifact.

Overfit probe (300 sequences, 100 epochs, no early stopping, lr grid
{1e-3, 3e-3, 1e-2} x seeds {0, 1} — fixed heads):

| model | min train MSE | test MSE |
|-------|---------------|----------|
| GRU (fixed) | **0.0154** (lr 1e-2, s1) | 0.1077 |
| MLP | 0.0290 | 0.1536 |
| DLinear | 0.0959 | 0.0983 |
| MMT (fixed) | 0.2173 – 0.2688 (all configs) | 0.2791 – 0.3122 |

(mean-predictor val reference 0.5679; the old MMT "plateau" was 0.496–0.509
against a broadcast floor of ~0.47.)  The toric model now trains well below
the old plateau, and a much smaller but real optimization gap remains
(train 0.22 vs 0.015–0.029 for the generic models).

### Item 2, hardened: seeds, shuffle control, full trees, direct fit

`ett_residual_lgbm.py` was extended: per-channel LightGBM on the frozen
DLinear-base residual is now run over 3 booster seeds with a shuffle control
(train targets row-permuted, val kept real), a no-early-stopping full-150-tree
variant (removes the val-selection bias of the 1–18-tree fits), residual R^2
on the never-touched test split, and a direct fit on the raw target (can
boosting beat the linear base at all?).

| quantity | value |
|----------|-------|
| residual R^2 (val, early stopping) | **+0.026 ± 0.001** (mean ± std, 3 booster seeds) |
| residual R^2 (test, untouched by selection) | **−0.006 ± 0.005** (≈ 0: the val gain does not transfer) |
| residual R^2 (val, 150 trees, no early stopping) | −0.127 ± 0.006 (ES was protecting, not underfitting) |
| shuffle control (val R^2) | +0.000 ± 0.001 (the noise floor) |
| direct LGBM on raw target, test MSE | 0.494 ± 0.001 (DLinear base: 0.390 — boosting loses by 27%) |
| base + LGBM (early stopping), test MSE | 0.385 ± 0.002 (a Δ ≈ 0.005 gain, consistent with the ≈0 test R^2) |

**Finding (rephrased per review):** LightGBM explains 2.6% ± 0.1% of the
DLinear residual on val (mean ± std over seeds; shuffle control 0.0% ± 0.1%),
but the same fits score −0.6% ± 0.5% on the never-touched test split — the
val-side gain does not transfer.  Fitting the full 150 trees without early
stopping makes val worse (−12.7%), so the 1–18-tree early stopping was a
regularizer, not a conservative bias.  Trained directly on the raw target,
boosting reaches test MSE 0.494 vs the linear base's 0.390.  The corrected
claim: **on this benchmark, at this feature family (raw lags + envelope
statistics + calendar phase), the per-channel linear base is the ceiling for
gradient boosting; the small val-side residual signal is a val-split effect,
and extrapolation structures of the residual are not covered by this test.**
Trees do not extrapolate.

### Control 1 — linear probes on the frozen MMT state (`probe_state.py`)

The trained MMT (Round-3 protocol, fixed head) is frozen; linear probes are
fit on four feature sets: the chain posterior `p_T`, the raw phase state
(48 angles, before fusion), the head input (what the forecast head actually
consumes), and the raw window.

Regime recovery, states=3 model (best-permutation agreement vs the true
regime; chance 0.33):

| probe | agreement |
|-------|-----------|
| logistic on raw window | 0.3517 |
| **model's own chain posterior, argmax** | **0.3633** |
| logistic on chain posterior | 0.4733 |
| logistic on head input | 0.9700 |
| **logistic on phase state** | **0.9867** |

Phase recovery (closed-form ridge on (cos theta_T, sin theta_T), true
generator phase): angle error 0.069 rad / cos-sin R^2 0.979 on the head
input; 0.045 rad / 0.992 on the raw window.

Linear readout forecaster (closed-form ridge, frozen features -> 12x2):
head input 0.3639 and state+posterior 0.3599 — both **below the trained
MMT head** (0.3878) and below ridge on the raw window (0.4237); on the
states=3 model the same holds (0.3803 vs head 0.4363).

Three consequences:

1. **The regime is almost perfectly linearly present in the phase state**
   (0.987 vs 0.35 for a linear map on the raw window — the regime evidence
   is deliberately nonlinear in the samples).  The interpretability claim
   must be re-anchored: it is the *continuous phase state*, not the discrete
   chain, that carries the regimes.  The chain posterior is barely better
   than chance (0.363 own-argmax; 0.473 re-weighted by a logistic probe).
2. **The state is better than its own head**: a single linear layer on the
   frozen representation beats the trained nonlinear head and beats any
   linear map on the raw window.  The encoder learned a genuinely useful
   representation; the output path still leaves ~6–15% on the table.
3. The Round-3 framing "the chain encodes the regimes but the head does not
   decode them" was right in spirit but wrong in mechanism: the *phase state*
   encodes them, the old head could not even address channels, and the
   discrete chain contributes almost nothing measurable.

### Control 2 — differentiable rotation, minimal probe (`phase_rnn.py`)

The Round-4 candidate in its cheapest form: a continuous unit-modulus state
advanced by input-conditioned learned angle kicks,

`h_{t+1} = normalize(h_t * exp(i * (theta_base + f(x_t))))`, `h_0 = 1`,

with f linear or a 2-layer MLP, no chain, no quantization, no attention —
everything differentiable — followed by the same fusion + linear head.
Same 300-sequence probe, same budget (100 epochs, no early stopping):

| model | min train MSE | test MSE |
|-------|---------------|----------|
| phase-RNN (mlp kick), lr 1e-3 | 0.0333 / 0.0357 | ~0.65 |
| phase-RNN (mlp kick), lr 3e-3 | 0.0333 – 0.0357 | ~0.65 |
| phase-RNN (mlp kick), lr 1e-2 | 0.1598 | 0.5589 |
| (old MMT broadcast plateau) | 0.496 – 0.509 | 0.513 – 0.530 |
| (fixed MMT, best config) | 0.2301 | 0.2791 |
| (MLP) | 0.0290 | 0.1536 |

The differentiable rotation **fully memorizes the 300 sequences** (train
0.033, 15.7 s per run) — the Round-3 "structural barrier of the phase-
recurrent parameterization" does not exist for the differentiable form; what
remains of the toric model's optimization gap is carried by its discrete /
quantized / gated machinery.  The probe also exposes the trade: with this
minimal recurrence the model overfits and generalizes poorly on the tiny
task (test ~0.65), and on the full regime task it reaches only 0.64–0.69
(the fixed MMT's chain/attention/gating is what buys 0.388 there).

## Verdict (after Round 3.5)

1. The Round-3 claims "the analytic phase readout without training beats the
   trained MMT (0.531 vs 0.896)" and "the toric model loses on a task built
   from its own assumptions" were **harness artifacts**: the trained model
   was structurally limited to the cross-channel mean.  With a corrected
   head, the trained toric model beats the analytic readout by 27% on the
   same task and sits within 4% of the MLP and 9% of the GRU.
2. The regime information lives in the **continuous phase state**, not in
   the discrete chain: a linear probe recovers the regime from the state at
   0.987 (chance 0.33; raw window 0.35; chain posterior 0.36–0.47), and a
   linear readout on the state beats the trained head.  The barrier that
   Round 3 attributed to "gradient descent on phase-recurrent dynamics" was
   an output-layer bug; what remains is a smaller, real optimization gap
   (probe train 0.22 vs 0.015–0.029), which a fully differentiable rotation
   removes entirely in the probe (train 0.033).
3. Hardened residual result: boosting over lag/envelope/calendar features
   explains 2.6% ± 0.1% of the post-linear ETTh1 residual on val (shuffle
   control 0.0% ± 0.1%) and ≈ 0% on test (−0.6% ± 0.5%); trained directly it
   cannot approach the linear base (0.494 vs 0.390).  The linear base is the
   ceiling **for this feature family**; extrapolation structures are not
   covered by this test.
4. Honest remaining gaps of the toric core: it still loses the 8-channel
   home-turf tasks to GRU/DLinear by 3–4x (optimization, not output cap);
   the discrete chain adds nothing measurable yet (posterior ≈ chance, and
   the 3-state variant forecasts worse than the 24-state one); on ETTh1 it
   remains far behind (1.087, a valid number).

## What to try next (revised after Round 3.5)

1. **Round 4 now has direct evidence behind it.**  The differentiable
   rotation passes the 2-minute overfit probe (train 0.033 vs the toric
   plateau 0.22); the next step is to combine the differentiable recurrence
   with the regime machinery (chain / geodesic gate / attention) and re-run
   the nonlinear task (target: beat 0.388) and the home-turf tasks (target:
   close the 0.385 vs 0.117 gap).  The 300-sequence probe stays the
   regression test: a candidate must fit materially below train ~0.1 before
   anything larger is worth running.
2. **The chain has to earn its place.**  Either feed the posterior to the
   head (the probes show state+posterior >= state alone) or admit the
   discrete state is dead weight; the 0.502 Round-3 agreement does not
   survive the fix (0.363), and the regime story belongs to the phase state.
3. **If the goal is forecasting accuracy on ETTh1**, the hardened Item 2
   bounds what any residual-correction approach can add at this feature
   family; a phase-based model must add value on the raw series, not on the
   post-linear residual.
4. The GPU-budget race is still gated on a parameterization fix passing the
   overfit probe; with the corrected harness, the probe now measures what it
   was always meant to measure.

## Round 4: plan and stop-loss (fixed before any run, per review)

Constraints fixed before launch, as demanded by the review:

- **Primary target is GRU's 0.3567** on the nonlinear task — not the toric
  model's own 0.3878.  Beating ourselves is not the goal.
- **Exactly 4 variants** on one shared differentiable rotation recurrence,
  run as an ablation ladder; every result is reported, including failures.
- **Home turf is the second gate**: the winner must also survive the task
  family it was not designed on.
- **The stop-loss below is fixed before launch**: no budget extensions, no
  post-hoc lr sweeps, no mid-flight architecture tweaks.
- **Rotation stays SO(2)**: the state lives on `(S^1)^d` and every transition
  is complex multiplication `exp(i * angle)`.  No quaternion (S^3)
  parameterization anywhere.

Shared base of all variants (the Round 3.5 control-2 recurrence):

`h_{t+1} = normalize(h_t * exp(i * (theta_base + f(x_t))))`, `h_0 = 1`,

`f` = 2-layer MLP kick, then the same fusion + linear head + amplitude
bypass + diff inputs.  The toric machinery is grafted back in pieces:

| variant | chain | geodesic gate | attention |
|---------|:-----:|:-------------:|:---------:|
| `rot` (reference) | – | – | – |
| `chain` | + | – | – |
| `gate` | – | + | – |
| `chain_gate` | + | + | – |
| `full` | + | + | + |

- **chain**: `ToricMarkovChain` (stochastic kernel, K=8) driven by the raw
  diff features; its posterior adds per-state kick offsets
  `state_theta[K, D]` to the rotation — the differentiable replacement of
  the toric state-selected kicks (`theta` init N(0, 0.5^2), as in the toric
  layer).
- **gate**: `g_t = sigmoid(Linear(x_t))` per angle dimension; the update
  becomes the torus geodesic slerp
  `normalize(g * h_rotated + (1 - g) * h_old)` — the toric gate made
  input-conditioned and per-dimension.
- **attention**: per-dimension pull toward a learned context phase (the
  toric attention minus quantization).

Fixed protocol, identical to the Round 3.5 tables: nonlinear task
2,500/600/600, T=64 H=12, 3 regimes, 30 epochs, patience 5, lr 2e-3,
batch 64, diff-input, amplitude, seed 0 then seed 1.  Home turf: easy =
6 epochs / patience 2 / plain features; hard = 25 epochs / patience 5 /
speed x2, T=96 H=24 / plain features (both exactly the protocols that
produced the reference numbers).

**Stop-loss (fixed):**

1. Pre-flight: the `full` variant must fit the 300-sequence probe to min
   train MSE < 0.10 within the 100-epoch budget.  Fail => nothing launches.
2. After seed 0 for all 5 configs (4 variants + reference, one fixed lr):
   if best test MSE > 0.388 (old MMT) => **STOP** — the direction is closed
   in its current form; no seed 1, no tuning, documented as failed.
3. If 0.357 < best <= 0.388: seed 1 runs for the best variant only
   (stability check); the verdict is "machinery helps but misses the
   target" regardless of the confirmation outcome.
4. If best <= 0.357: seed 1 for the best two variants; the winner (selected
   on **val**, never on test) goes to the home-turf gate.
5. Secondary gate: **pass** = beats old MMT on both home-turf tasks
   (0.3852 easy, 0.4815 hard); **strong pass** = within 1.5x GRU
   (easy <= 0.176, hard <= 0.393); failure on either task => the machinery
   is task-specific and is reported as such.

### Round 4 outcome: pre-flight gate FAILED — nothing launched

The pre-flight was run exactly as registered (`phase_hybrid.py`, committed
before any run; probe protocol identical to the Round 3.5 control: 300
sequences, 100 epochs, no early stopping, plain features, seed 7 data).

| variant | min train MSE (100 ep) | probe test MSE | pre-flight (< 0.10) |
|---------|------------------------|----------------|---------------------|
| rot | 0.0357 (lr 3e-3) | 0.6511 | (reference, passes) |
| chain | 0.0361 | 0.6299 | passes (diagnostic) |
| gate | 0.0294 | 0.5945 | passes (diagnostic) |
| chain_gate | 0.0305 | 0.6181 | passes (diagnostic) |
| **full** | **0.1114 / 0.1124 / 0.1230** (lr 3e-3 / 1e-2 / 1e-3) | **0.1316** | **FAILS** |

Per the fixed stop-loss, the primary gate did **not** launch: no nonlinear
runs, no home-turf runs, no seed 1.  The post-mortem diagnostics (cheap
probes of the remaining ladder, same budget) localize the failure:

1. **The blocker is the attention pull.**  `chain` and `gate` are
   optimization-neutral (train 0.029-0.036, indistinguishable from `rot`),
   and `chain_gate` fits fine (0.0305).  Adding attention to `chain_gate`
   multiplies the probe train MSE by 3.7x (0.0305 -> 0.1114), consistently
   across all three learning rates.  The Round-3.5 claim "the remaining
   optimization gap is carried by the discrete / quantized / gated
   machinery" must be corrected: quantization-free chain and gate do NOT
   impede fitting — the context-phase pull does.
2. **The same attention is the strongest regularizer seen so far.**  With
   attention, the probe test MSE drops from ~0.59-0.65 (every other
   variant, all of which memorize the 300 sequences) to **0.1316** — a 4.5x
   generalization improvement, while train stays at 0.11.  The context
   pull constrains the dynamics enough to prevent memorization entirely.
3. For calibration: `full` at 0.111 still fits the probe **2x better than
   the fixed MMT itself** (train 0.217-0.269) while carrying the same
   machinery plus attention.  The 0.10 threshold was calibrated against
   the bare differentiable recurrences (0.03), not against
   machinery-carrying models.

Status: **Round 4 is closed as pre-registered.**  Any continuation (a
ladder without attention, or a re-justified criterion for attention-carrying
variants) must be re-registered as Round 4b before any further run — that
is a decision to be approved, not slipped through.

## Round 4b: plan and stop-loss (re-registered before any run)

Decision after the pre-flight failure, fixed by review: **option A** — the
ladder without attention.  The reviewer's rationale, recorded because it
binds this and every future round: a criterion that failed must not be
renegotiated after the fact; the value of pre-registration is precisely
that it cannot be revised retroactively — otherwise, six months later (or
at review), "plan followed" and "plan bent around the result" become
indistinguishable.

**Protocol success, recorded explicitly:** the Round-4 stop-loss worked as
designed.  A pre-flight that blocks a launch is the protocol succeeding,
not Round 4 failing.

**Findings carried into the record independently of Round 4b's outcome:**

1. **Attention finding (separate item, not a Round-4 component):**
   attention acts as a strong regularizer on the synthetic probe — train
   x3.7 (0.0305 -> 0.1114), test 0.62 -> 0.13; it blocks overfitting by
   construction.
2. **Calibration fact:** full 0.111 vs rot 0.036 — the context pull
   constrains fitting ~3x.  True whether or not attention ships.
3. **Open question, reserved for a separate pre-registration:** can the
   attention regularizer learn the real-task signal (2,500 sequences, a
   different regime than the 300-sequence probe)?  A model that cannot
   overfit the probe has no demonstrated ability to fit a real task.  New
   hypothesis, new gate — not a revision of Round 4.

**Ladder (all passed the registered 0.10 pre-flight):** rot 0.0357,
chain 0.0361, gate 0.0294, chain_gate 0.0305.

**One new hypothesis, registered here, same 0.10 bar:** `gate_attn_low` —
gate + attention with a *fixed 0.1x pull coefficient* (suspected culprit:
the pull's scale, not the mechanism).  It must pass the same pre-flight
(probe, 100 epochs, min train MSE < 0.10) before joining the ladder.  If
it passes and wins: regularization without the optimization blockage.  If
it fails either gate: attention is disqualified as a component and remains
a finding.  This is "add a hypothesis", not "change the criterion".

**Fixed protocol:** nonlinear task, identical to Round 3.5 (2,500/600/600,
T=64 H=12, 3 regimes, 30 epochs, patience 5, lr 2e-3, batch 64,
diff-input, amplitude), **5 seeds {0,1,2,3,4} per variant**, mean +/- std
reported; no per-variant tuning of any kind.  Winner = lowest *mean val*
MSE, never test.

**Stop-loss (fixed):**

1. `gate_attn_low` pre-flight: probe min train < 0.10, else it is out of
   the ladder and attention is disqualified as a component.
2. Primary outcomes on the nonlinear task (test MSE, **mean over 5
   seeds**):
   - **rot alone <= 0.357 (GRU):** chain/gate are dead weight — "the toric
     machinery is dead, the rotation is alive".  A valuable result.
   - **only with chain and/or gate <= 0.357:** the components work;
     decompose which one and interpret.
   - **nothing <= 0.373 (MLP):** the project is closed for real, without
     "almost".  (0.357 < mean <= 0.373: beats the MLP, misses the target —
     reported as improvement, not victory.)
3. Secondary gate (home turf, winner by mean val, 5 seeds, easy + hard at
   the exact reference protocols): pass = beats old MMT on both tasks
   (0.3852 easy, 0.4815 hard).  If the nonlinear winner does not transfer,
   the README says exactly that — "success on nonlinear does not transfer
   to home turf" — an open question, not a victory.
4. Budget: up to 5 configs x 5 seeds (nonlinear) + 2 tasks x 5 seeds
   (winner) + 3 probe runs (gate_attn_low).  Nothing beyond this; no
   extensions, no tuning.

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

# Round 3 (roadmap items 1-3: nonlinear task, LightGBM residual, long-budget probe)
python -m research.nonlinear_torus --models dlinear,mlp,moe_linear,gru,mmt,naive,phase \
    --epochs 30 --patience 5 --diff-input --amplitude --seed 0 \
    --out research/results_nonlinear.json
python -m research.nonlinear_torus --models mmt --states 3 --epochs 30 --patience 5 \
    --diff-input --amplitude --seed 0 --out research/results_nonlinear_states3.json
python -m research.ett_residual_lgbm --base-epochs 8 \
    --out research/results_ett_residual_lgbm.json
python -m research.overfit_budget --models mmt,gru,mlp,dlinear \
    --lrs 1e-3,3e-3,1e-2 --seeds 0,1 --epochs 200 --patience 50 \
    --out research/results_overfit_budget.json

# Round 3.5 (harness fix re-runs + pre-Round-4 controls + hardened item 2)
python -m research.nonlinear_torus --models mmt --epochs 30 --patience 5 \
    --diff-input --amplitude --seed 0 --out research/results_nonlinear_fixed.json
python -m research.nonlinear_torus --models mmt --states 3 --epochs 30 --patience 5 \
    --diff-input --amplitude --seed 0 --out research/results_nonlinear_states3_fixed.json
python -m research.nonlinear_torus --models gru --epochs 30 --patience 5 \
    --diff-input --amplitude --seed 0 --out research/results_nonlinear_fixed.json
python -m research.synthetic_torus --models mmt,gru --epochs 6 --patience 2 \
    --out research/results_synthetic_fixed.json
python -m research.synthetic_torus --models mmt --epochs 25 --patience 5 \
    --speed-scale 2.0 --horizon 24 --seq 96 --out research/results_synthetic_hard_fixed_mmt.json
python -m research.synthetic_torus --models gru --epochs 25 --patience 5 \
    --speed-scale 2.0 --horizon 24 --seq 96 --out research/results_synthetic_hard_fixed_gru.json
python -m research.overfit_budget --models mmt \
    --lrs 1e-3,3e-3,1e-2 --seeds 0,1 --epochs 100 --patience 100 \
    --out research/results_ob_fixed_a.json   # plus --lrs 1e-2 into _b.json
python -m research.overfit_budget --models gru \
    --lrs 1e-3,3e-3,1e-2 --seeds 0,1 --epochs 100 --patience 100 \
    --out research/results_ob_fixed_gru.json
python -m research.probe_state --states 24 --epochs 30 --patience 5 \
    --diff-input --amplitude --seed 0 --out research/results_probe_states24.json
python -m research.probe_state --states 3 --epochs 30 --patience 5 \
    --diff-input --amplitude --seed 0 --out research/results_probe_states3.json
python -m research.phase_rnn --task probe --kick mlp --lrs 1e-3,3e-3 --seeds 0,1 \
    --epochs 100 --out research/results_phase_rnn_probe_a.json   # plus --lrs 1e-2 into _b.json
python -m research.phase_rnn --task nonlinear --kick mlp --epochs 30 --patience 5 \
    --diff-input --amplitude --seed 0 --out research/results_phase_rnn_nonlinear.json
python -m research.phase_rnn --task nonlinear --kick linear --epochs 30 --patience 5 \
    --diff-input --amplitude --seed 0 --out research/results_phase_rnn_nonlinear_linear.json
python -m research.ett_residual_lgbm --base-epochs 8 --seeds 0,1,2 \
    --stride-train 3 --stride-val 6 --out research/results_ett_residual_lgbm.json

# Round 4 (pre-flight + post-mortem probes only; the primary gate was NOT launched)
python -m research.phase_hybrid --task probe --variant full --lrs 1e-3,3e-3,1e-2 --seeds 0 \
    --epochs 100 --out research/results_round4_probe_full.json
python -m research.phase_hybrid --task probe --variant rot --lrs 3e-3 --seeds 0 \
    --epochs 100 --out research/results_round4_probe_rot.json
python -m research.phase_hybrid --task probe --variant chain --lrs 3e-3 --seeds 0 \
    --epochs 100 --out research/results_round4_probe_chain.json
python -m research.phase_hybrid --task probe --variant gate --lrs 3e-3 --seeds 0 \
    --epochs 100 --out research/results_round4_probe_gate.json
python -m research.phase_hybrid --task probe --variant chain_gate --lrs 3e-3 --seeds 0 \
    --epochs 100 --out research/results_round4_probe_chain_gate.json
```

Result JSON files are committed next to this README.
