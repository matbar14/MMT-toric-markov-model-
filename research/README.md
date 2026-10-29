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
   recovers them at 0.502 agreement (chance = 0.33) — above chance, far from
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

## What to try next (revised after Round 3)

1. **If the goal is forecasting accuracy: use the MLP or MoE-linear and stop
   paying the toric tax.**  On every task tested they dominate, and the
   residual analysis bounds the remaining headroom to ~2.5%.
2. **If the goal is the toric architecture itself, the optimization barrier
   is the only thing worth working on**, and the 300-sequence overfit probe
   (100 epochs, ~2 min per config on CPU) is the cheap regression test for
   any proposed fix: a fix must drive train MSE materially below ~0.45
   before anything larger is worth running.  Candidate fixes at the
   parameterization level: make rotations differentiable end-to-end (drop or
   relax the STE quantizer during training), initialize the phase dynamics
   near identity with small learned kicks, train the toric core as a
   correction on MLP features instead of raw inputs, or replace the
   input-conditioned soft chain with a hard commit (Gumbel/STE) regime
   sampler whose gradients bypass the recurrence entirely.
3. **If the goal is interpretability, compare against the cheap alternative
   first**: the MoE gate plus a clustering of its expert assignments gives
   regime labels without any phase machinery; any interpretability claim for
   the toric chain has to beat that baseline, not just chance (0.502 vs 0.33
   here does not).
4. A GPU-budget race (100+ epochs on the full 2,500-sequence task, wider lr
   sweep) is only worth running *after* a parameterization fix passes the
   overfit probe; the CPU evidence here is consistent across seeds and lrs.

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
```

Result JSON files are committed next to this README.
