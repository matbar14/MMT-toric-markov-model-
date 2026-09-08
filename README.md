# Toric Markov — Experimental Trading Research

> **Experimental / research-only. Not production-ready. No demonstrated profitable edge.**
> This repository contains models, historical simulators and reproducible experiments,
> not a ready-to-use trading bot. No live order execution is implemented. Backtests,
> passing tests and accurate forecasts do not guarantee future returns.

**По-русски:** это экспериментальная исследовательская модель, а не готовый
торговый продукт. Прибыльность не подтверждена. В текущем тесте дельта-нейтральной
стратегии все три временных разбиения выбрали отсутствие торговли. Реальные
ордера не отправляются; результаты публикуются вместе с ограничениями и убытками
диагностических стратегий, без обещаний заработка.

## Research Status

- **Current focus:** cost-aware BTCUSDT spot/perpetual paired-return research,
  with chronological training, selection, calibration and evaluation.
- **Latest recorded experiment:** the September 2025–September 2026 dataset;
  all three folds select cash (zero trades, zero return). None of 1,085 scheduled
  six-hour pairs is profitable after the configured costs.
- **Diagnostic, not model performance:** forcing every pair to trade loses
  80.675% of starting capital (92.883% under stressed costs). The actual selected
  policy does not trade. See `ARBITRAGE_PROFITABILITY_REPORT.md` for assumptions.
- **Limitations:** previously inspected single-symbol history, hourly rather
  than executable bid/ask prices, assumed account fees, no production margin
  engine, no handling of partial fills or exchange outages.
- **Scope:** earlier directional experiments are retained for comparison.
  They are not advertised as working strategies. The optional C++ trainer
  covers the legacy V3 task, not the new paired-return objective.

Research results are not investment advice. Do not treat this repository as
a signal service, a risk-free arbitrage system or evidence of future profitability.

## Project Overview

PyTorch trading research with a shared complex-valued recurrent encoder.
The new delta-neutral research path predicts spot/perpetual paired returns;
the legacy directional path retains its event gate and 16 pattern labels.
Package version is **0.2.0**. Legacy checkpoints use format **2**; paired-return
checkpoints use the separate **toric_arbitrage_v1** format and require retraining.
Correctness tests do not establish predictive quality or profitability.

The year-data hold-collapse investigation, reproduced results and production
limitations are recorded in `MODEL_REPAIR_REPORT.md`. The repaired model emits
signals, but the evaluated strategy loses money after costs; it is not approved
for live trading.

## Delta-Neutral Research

The current redesign is **long BTCUSDT spot + short the same BTC quantity in
the linear USDT perpetual**, not an equal-dollar long/short approximation.
See `ARBITRAGE_REPORT.md`. This is a basis/funding research strategy, not a
risk-free or production-ready arbitrage bot. Existing directional experiments
remain available for reproducibility; their weights are not reused here.

The first real-history run is documented in `ARBITRAGE_PROFITABILITY_REPORT.md`.
All three chronological folds select cash. At the fixed six-hour settings and
assumed costs, none of 1,085 scheduled pairs has positive realized net payoff.
The diagnostic always-enter portfolio loses 80.675%; this is **not** the model's
realized loss, since the model does not trade. No profitable edge is established.

- `ToricArbitrageModel` predicts net paired PnL divided by committed spot cash
  plus short collateral, rather than BTC direction. Features describe basis,
  mark/trade differences and only previously settled funding.
- Both legs pay entry and exit fees and adverse slippage. Funding is booked
  at actual settlement events using the event mark price, never hourly
  interpolation. Ambiguous boundary receipts are omitted; debits are retained.
- Intraday slots are **00/07/14 UTC**, with six-hour holding periods ending
  at **06/13/20 UTC**. Idle gaps keep funding-boundary guards disjoint.
- Initial short collateral is at least 100% of its opening notional. A
  conservative mark-high collateral buffer check fails closed; it is not a
  replacement for exchange liquidation tiers or a real margin engine.
- The fixed chronological comparison includes Toric, Ridge, a cost-aware
  hypothetical basis-convergence rule, a last-settled-funding carry rule, and cash. Selection, calibration and
  evaluation are separate, with frozen weights/policies and stressed costs.
- The downloader requires the original BTCUSDT `market.csv` and matching
  `manifest.json`, adds mark/funding checkpoints, and verifies input hashes.
  Missing history is an error, not zero funding or a trade-price mark proxy.

```bash
.venv/bin/python toric_markov_model/scripts/download_market_year.py \
  --symbol BTCUSDT --days 365 --end 2026-09-07T22:00:00Z \
  --output-dir data/btcusdt_1h_20250907_20260907 --resume

.venv/bin/python toric_markov_model/scripts/download_arbitrage_inputs.py \
  --market-data data/btcusdt_1h_20250907_20260907/market.csv \
  --output-dir data/btcusdt_1h_20250907_20260907/arbitrage --resume

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python toric_markov_model/scripts/research_arbitrage.py \
  --market-data data/btcusdt_1h_20250907_20260907/market.csv \
  --arbitrage-data data/btcusdt_1h_20250907_20260907/arbitrage \
  --output-dir checkpoints_arbitrage_new
```

Default spot/futures fees of 0.10%/0.05% and per-leg slippage of 0.02% are
research assumptions **per fill**, not verified account rates. Override
`--spot-fee`, `--futures-fee`, `--spot-slippage` and `--futures-slippage` for
the intended account/execution assumptions. There is no order submission,
partial-fill handling or live inference service in this research path.

## Intraday Research

The earlier directional signal experiment is intraday, not weekly. See `INTRADAY_REPORT.md`.
It uses hourly data, long/flat spot entries at 00/06/12/18 UTC, a six-hour maximum
holding time and no positions carried beyond the UTC day. Three fixed causal
signals are compared before attempting a Toric entry filter; greater trading
frequency is not treated as evidence of an edge.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python toric_markov_model/scripts/research_intraday.py \
  --data data/btcusdt_1h_20250907_20260907/market.csv \
  --output-dir checkpoints_intraday_new
```

The fixed research assumptions are 2% stop, 3% take, 20% position fraction,
0.10% fee and 0.02% adverse slippage **per side**. These are not recommended or
optimized trading settings. Calibration must pass net/stressed costs and trade
support (at least 30 trades and 0.5 trades/day). Only then are long-only Toric
and Ridge filters trained on earlier entries; filter acceptance uses paired
daily returns, retained trade count and stressed costs. Failed calibration
disables the policy but does not hide its raw evaluation losses. No live orders
are submitted. The inspected one-year dataset remains development data.

### Resumable Market Download

The hourly downloader saves validated pages in per-market JSON checkpoints and
retries missing hours separately. `--resume` requires the same symbol, days and
end time, or reuses the saved end time when `--end` is omitted. Completed datasets
are never overwritten. Persistent gaps are listed in `*_checkpoint.gaps.json`;
no OHLC interpolation or incomplete `market.csv` is permitted.

```bash
.venv/bin/python toric_markov_model/scripts/download_market_year.py \
  --symbol BTCUSDT --days 1461 --end 2025-09-07T22:00:00Z \
  --output-dir data/btcusdt_1h_20210907_20250907 --resume
```

`--resume` also permits a fresh empty directory. Downloads made by the old
memory-only version cannot be recovered if the process exited before saving.

## Strategy And Stop Research

`scripts/research_strategies.py` compares the old causal CVD/basis rules with
three alternative entry families: trend breakout, trend pullback and range
reversion. The experiment is described in `STRATEGY_STOP_REPORT.md`.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python toric_markov_model/scripts/research_strategies.py \
  --data data/btcusdt_1h_20250907_20260907/market.csv \
  --output-dir checkpoints_strategy_new \
  --legacy-trades checkpoints_btc_year_run2/test_backtest_trades.csv
```

- The fixed 48-variant grid combines four entry families, four exit settings and
  4/12/24-bar holding limits. A 2% fixed stop keeps the same 2% take-profit as the
  1% control, isolating stop widening. ATR stops use 2 or 3 times ATR14/previous
  close, and a 4-ATR-normalized take. These percentages are frozen at entry;
  they do not trail, and absolute distances scale with the actual fill price.
- ATR uses Wilder smoothing with an arithmetic-mean seed. Signals, regime filters
  and volatility estimates use completed candles only; entries occur next open.
  Indicators may use causal history before a segment, but its first signal bar
  and all trade exits remain within that segment.
- The shared simulator supports per-entry stop/take distances and an optional
  planned stop-risk budget. Default sizing is capped at 20% of equity and targets
  at most 0.2% equity loss at the modeled stop, including fees and exit slippage.
  Wider stops reduce size; gaps can still exceed this budget. Fixed 20% sizing
  is reported alongside it to distinguish stop effects from reduced exposure.
- A candidate is selected only on the selection interval. The subsequent
  calibration interval can reject it; there is no fallback search there. The
  frozen policy is hashed before evaluation. The full diagnostic grid is also
  reported, but its hindsight winner is not deployed.
- Every horizon uses the same entry opportunity range and end date, with idle
  equity padded after shorter horizons. Cost stress uses 1.5x fees and 2x slippage.
  Funding, borrowing, impact and sub-hour intrabar paths are not modeled.
- Optional `--legacy-trades` replays the paired V3 ledger under its original
  1% stop / 2% take / 4h / 0.1%-per-side fee / zero-slippage assumptions. It
  rejects a mismatching ledger rather than silently analyzing different trades.
  Same-entry stop counterfactuals are isolated trades, not an executable portfolio.

This rule-strategy study does **not** retrain or silently change the payoff model,
its labels, legacy checkpoints or live settings. All outputs are research only;
no trading orders are submitted. The studied year has already been inspected.

## Cost-Aware Payoff Research

The new experimental path tests whether the inputs predict a **net trading
payoff**, rather than another heuristic pattern label. See
`PAYOFF_RESEARCH_REPORT.md` for the real-data experiment and its negative result.
It does not replace the legacy classifier or reinterpret its checkpoints.

- `ToricEncoder` is shared with V3, preserving its parameter names and C++ parity.
  `ToricPayoffModel` defaults to two regression outputs: expected net long/short
  returns. The intraday spot filter uses one long-only output. Targets and
  normalizers are fitted only on the training segment.
- Targets and execution use the same next-open / stop-first / time-exit rules,
  with entry/exit fees and adverse slippage. Funding and borrowing are not modeled.
- Features use lagged returns, volume, basis and volume-normalized CVD changes.
  Twelve CVD/basis conditions are known-at-close inputs, not future-filtered labels.
  Missing OI and the approximate POC rules are omitted in this experimental path.
- Three expanding-window folds separate training, checkpoint/architecture
  selection, policy calibration and evaluation. Context windows and target
  endpoints stay inside each segment; rolling indicators use causal prior history.
- Fixed ridge, histogram-gradient-boosting and train-mean controls share the same
  targets. Raw zero-edge and rule-only results are always reported, so a disabled
  policy cannot hide an unprofitable model behind a zero-return backtest.
- Calibration screens ten policies: five minimum expected net returns, with/without
  a causal pattern filter. Minimum trade support, signal-rate limit, positive
  calendar halves and a block-bootstrap screen may reject all policies.
  These are research screens, **not a statistical certificate or production approval**.

Run from the repository root with the installed project:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python toric_markov_model/scripts/research_payoff.py \
  --data data/btcusdt_1h_20250907_20260907/market.csv \
  --output-dir checkpoints_payoff_new --epochs 12 --folds 3 --seq-len 16
```

Every run requires a new output directory. The CLI writes a protocol before
training, fold reports, forecasts, trade ledgers, locally trusted baseline
artifacts and restricted-load `toric_payoff_v1` checkpoints. Baseline `.joblib`
files use pickle: never load files from untrusted sources. The new payoff trainer
currently runs in Python/PyTorch; the C++ trainer still trains the legacy V3 task.

Inspect a saved Toric forecast using **completed** candles:

```bash
python toric_markov_model/scripts/predict_payoff.py \
  --data data/btcusdt_1h_20250907_20260907/market.csv \
  --checkpoint checkpoints_payoff_new/fold_3/toric_payoff.pt
```

This reports expected net returns and a **paper-only** action. If Toric was not
selected or its policy failed calibration, the action is HOLD. It validates the
feature schema, bar interval, train statistics and calibration end date, but
the caller is responsible for supplying only completed candles. It never sends
orders; `production_approved` and `live_orders_allowed` remain false. The already
inspected year is developmental evidence, not a new independent holdout.

## Install

Python 3.10+ is required. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ./toric_markov_model
```

The CPU wheel is sufficient for the test suite. Use `--device cuda` only with
a working CUDA installation; an unavailable requested device raises an error.
`setup.py` declares runtime dependencies; `toric_markov_model/requirements.txt`
lists the same dependencies for environments managed separately.

## Repository Contents

The publishable project consists of source code, tests, build instructions,
the MIT license and research reports. A fresh clone is not an already trained
model: create the environment, download and validate the inputs, then run the
documented experiment. Download commands use public market endpoints and do
not need trading API keys; access may depend on the network and jurisdiction.

Market CSV files, local `.venv`, trained weights, generated build files and
checkpoint/run directories are local artifacts, not required source code.
Paths such as `checkpoints_arbitrage_real_v1/report.json` in the reports describe
the author's recorded run; those local artifacts are not supplied with a fresh
clone. Input hashes and settings are documented, but dependency versions and
hardware are not locked, so an independent rerun is not promised to be bitwise
identical. Never commit private keys, API credentials or account data.

## Data And Target Contract

CSV rows must have unique, regularly spaced timestamps, finite OHLC/CVD/basis
values and positive prices and volumes. Missing bars and duplicate timestamps
are rejected rather than silently changing the meaning of the horizon.
Spot and futures columns are required; open interest is optional.

For an entry at row `t`, a sequence length `L` and holding horizon `H`:

- Inputs are rows `[t-L, t)`, ending at the previous completed candle.
- Pattern conditions use data through `t-1`, never the entry candle.
- Entry is at `spot_open[t]`; the target exit is `spot_close[t+H-1]`.
- The return label is `exit / entry - 1`, before transaction costs.
- Volume, CVD and POC targets compare the horizon endpoint to the previous
  completed candle. Their outputs are standardized during training only.
- Samples without a complete horizon inside their split are excluded. The
  final valid sample has a real endpoint; missing targets are not filled with zero.

Labels remain heuristic pattern conditions filtered by a future gross-return
threshold, not guarantees of executable trades. Backtest stops, fees and fills
must be evaluated separately from classification quality.

Features are computed causally before splitting, preserving rolling history.
Default splits are chronological **70% train / 15% validation / 15% test**.
Window context and label endpoints stay inside each selected segment.
Normalization is fitted only on training data; auxiliary statistics use only
valid training targets. Evaluation requires those saved statistics.
Exact split dates and ordered feature names are stored in the checkpoint.
Appending history does not move the saved train/validation boundaries.

## Train

From the repository root with `.venv` activated:

```bash
python toric_markov_model/scripts/train_trading_v3_basis.py \
    --data /path/to/market.csv \
    --checkpoint-dir checkpoints_v3 \
    --epochs 10 --batch-size 32 --device cpu
```

The trainer uses weighted BCE for the gate and conditional patterns, and
Smooth L1 for standardized auxiliary targets. It does not combine class
weights with a balanced sampler. It retains the final partial batch and
reports per-pattern counts, precision, recall and F1 using the same decoding
rules as inference. Reported losses are aggregated by their relevant sample counts.
Training stops on nonfinite loss or gradients.

- `--stage 0`: joint training; checkpoint selection minimizes joint validation loss.
- `--stage 1`: encoder and gate training; selection minimizes validation gate loss.
- `--stage 2`: pattern-head training with a frozen encoder/gate; requires
  `--resume-from` and identical model/data settings. Hold-only batches do not
  call the optimizer, preventing both backward errors and unintended weight decay.

All stages select checkpoints and reduce the learning rate using loss, not F1
at an arbitrary 0.5 threshold. This prevents early stopping at the first epoch
when scores are still below the decision threshold. Stage 2 selects by conditional
pattern loss. Validation is divided chronologically into selection and calibration
halves (`--calibration-fraction 0.5`); their windows and target horizons do not
overlap. Calibration is not passed to either trainer.

After selecting a checkpoint, fit decision thresholds on the reserved tail:

```bash
python toric_markov_model/scripts/calibrate_trading.py \
    --data /path/to/market.csv \
    --checkpoint checkpoints_v3/best_model_stage0.pt \
    --output checkpoints_v3/calibrated_model_stage0.pt
```

The fixed 81-pair grid maximizes strongest-pattern F1, requires at least 20
signals, a signal rate no higher than 50% (`--max-signal-rate`), event precision
above event prevalence, and improvement over an
always-most-common-pattern baseline. If none qualify, it writes a diagnostic
JSON but **no threshold-tuned checkpoint**. This is decision-threshold fitting, not
probability calibration or evidence of profitability. Both always-hold and
nearly-always-active degeneration are rejected; this frequency constraint is
a policy, not proof of a trading edge. Reports explicitly mark production
approval false. The backtest reads the
saved thresholds automatically. Direct `load_checkpoint()` / `model.detect_patterns()`
inference also uses them; explicitly supplied thresholds override them. Never tune
thresholds against test profit or signal count. A test already inspected during
development is diagnostic, not a fresh
out-of-sample confirmation. Warm-start cannot reuse a calibrated checkpoint or
change the reserved partition.

For a reproducible diagnostic comparison with train-only logistic-regression
and histogram-gradient-boosting event gates:

```bash
python toric_markov_model/scripts/diagnose_trading.py \
    --data /path/to/market.csv \
    --checkpoint checkpoints_v3/calibrated_model_stage0.pt \
    --output checkpoints_v3/diagnostics.json
```

It reports selection/calibration AP, ROC AUC, event prevalence, signal counts
and strongest-pattern quality. `--include-test` explicitly includes the
historical test as a diagnostic; this command never searches thresholds.

`--resume-from` is an explicit **warm-start**, not exact optimizer/RNG resume.
Checkpoints include configuration, state, training statistics, data SHA-256,
feature schema, decision thresholds, optimizer/scheduler state and validation
metrics. Writes are atomic. Only training and validation are used by this CLI.

## Native C++ Training

A separate LibTorch trainer is available in `toric_markov_model/cpp`.
It keeps CSV preprocessing in Python, then executes forward/backward and
optimization in C++, and converts results to the existing checkpoint format.
See `toric_markov_model/cpp/README.md` for CMake build commands, training,
numerical parity tests and a reproducible Python/C++ CPU benchmark.
Acceleration is workload-dependent; CUDA performance has not been verified.

## Model Outputs

The encoder uses the same deterministic soft state transitions in training
and evaluation. Dropout is disabled by `eval()` as usual. The context module
changes the complex state direction rather than multiplying by a scalar that
is immediately normalized away. Redundant magnitude controls and untrained
regime/reversal/breakout heads have been removed.

The pattern head predicts 16 labels conditional on an event. Hold probability
is derived as `1 - gate`, not produced by an untrained seventeenth classifier.
The compatibility field `pattern_confidence` is a joint score computed as
`gate * conditional_pattern_score`. There is no duplicate confidence head,
and these scores are **not claimed to be calibrated probabilities**.

`gate_threshold`, `pattern_prob_threshold` and `confidence_threshold` filter
the event gate, conditional pattern score and joint score respectively. Active
patterns, strongest pattern, hold and `has_pattern` use one shared decision rule.
`detect_patterns()` requires `model.eval()` and saved auxiliary statistics;
it returns regressions in physical return/change units, not z-scores.

## Evaluate

```bash
python toric_markov_model/scripts/backtest_trading_v3.py \
    --data /path/to/market.csv \
    --checkpoint checkpoints_v3/best_model_stage0.pt \
    --output backtest_test.csv --device cpu
```

By default this evaluates the held-out **test** segment using the checkpoint's
fixed boundaries and thresholds. Supply the full history so rolling features
can be reconstructed. `--max-hold-bars` defaults to the training horizon and
counts the entry candle. TP/SL are checked on the entry candle too, gaps fill
at the opening price, and remaining tail bars can close positions without
opening entries whose horizon would be incomplete.

Threshold grid search is allowed only with `--split validation`; attempting
`--optimize-signal-threshold` on test raises an error. Freeze validation-selected
settings before the final test. Repeated manual tuning on test invalidates its
role as an independent evaluation.

```bash
python toric_markov_model/scripts/analyze_confidence.py \
    --data /path/to/market.csv \
    --checkpoint checkpoints_v3/best_model_stage0.pt --device cpu
```

Confidence diagnostics use **validation**, not test. Experimental external
gate scripts and the enhanced dataset are retained, but external gate artifacts
are not accepted in the final test CLI because their training provenance is not
verified. Old filtered NPZs, metrics and gate weights are historical artifacts,
not validation evidence for the current model.

## Compatibility

**Existing checkpoints require retraining.** They use a different architecture,
label contract and output scaling. Loading format 1, silently using
`strict=False`, or falling back to validation-fitted normalization is unsupported.
The new loader uses restricted tensor/dictionary loading and strict state matching.

Legacy token models, their cache/tokenizer/trainer/configs and V1/V2 trading
datasets remain removed. Obsolete training flags for confidence losses, balanced
sampling and the previous stage-specific loss variants are no longer accepted.

## Validation

From the repository root:

```bash
python -m unittest discover -s toric_markov_model/tests -t toric_markov_model -v
```

Tests cover dataset endpoints and causality, split separation and normalization,
Markov mode consistency, attention gradients, decision consistency, output
scaling, hold-only batches, last-batch metrics, strict checkpoint loading,
packaging discovery and entry-candle exits. A synthetic CLI integration test
trains one epoch in each stage, reloads checkpoints, runs the held-out backtest
and verifies that threshold optimization on test is rejected.

These are correctness checks, not market benchmarks. Real-data walk-forward
evaluation, realistic execution assumptions and operational monitoring remain
necessary before live use. This repository does not implement live order execution.

## License

MIT.
