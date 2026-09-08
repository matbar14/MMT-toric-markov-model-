# Native LibTorch trainer

The executable implements the current V3 forward pass, straight-through angle
quantization, backward pass, gradient clipping, AdamW, train/validation metrics,
plateau LR reduction and early stopping in C++. There are no Python calls inside
an epoch. CSV feature extraction and labeling remain in the existing Python
dataset, avoiding a second incompatible implementation of preprocessing.

## Build

Run from the repository root after installing the Python project into `.venv`:

```bash
source .venv/bin/activate
cmake -S toric_markov_model/cpp -B toric_markov_model/cpp/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build toric_markov_model/cpp/build -j 1
```

Requires CMake 3.18+ and a C++20 compiler. Using LibTorch from the installed
PyTorch package keeps headers, libraries and ABI flags aligned. Rebuild after
changing the installed PyTorch version. Builds can use substantial RAM; `-j 1`
limits concurrent compiler processes. Tested locally with GCC 13.3 and CPU
PyTorch 2.14.0. CUDA requires a CUDA-enabled LibTorch build and working driver;
the local verification covers CPU only.

## Train

```bash
python toric_markov_model/scripts/train_trading_cpp.py \
  --data /path/to/market.csv \
  --checkpoint-dir checkpoints_cpp_run1 \
  --epochs 10 --batch-size 32 --threads 1 --device cpu
```

Use a new checkpoint directory per run. Supported options share the existing
Python trainer's data, architecture and loss settings; `--help` lists them.
`--stage 0`, `--stage 1` and `--stage 2` are supported. Stage 2 needs a compatible
checkpoint through `--resume-from`; both Python and converted C++ checkpoints
can be used. This is a weights warm-start, not an exact optimizer/RNG resume.
Native runs do not create a Python-compatible optimizer state.

The launcher performs three steps:

1. Construct train and validation with the same boundaries, features, labels
   and training-only statistics as the Python trainer. No test data is exported.
2. Run `cpp/build/toric_train` on those prepared tensors. Windows are strided
   views of the feature matrix; only selected minibatches are materialized.
3. Convert best/last native weights to checkpoint format 2, retaining model
   configuration, data hash, feature names, split dates and normalization.

Result files include:

- `best_model_stage0.pt`, `last_model_stage0.pt`: ordinary Python checkpoints
  usable by the existing backtest loader; filenames follow the requested stage.
- `native/input.pt`: initial parameters, compact datasets and loss settings.
- `native/template.pt`: Python checkpoint metadata and normalization.
- `native/best_weights.pt`, `native/last_weights.pt`: native parameter archives.
- `native/metrics.jsonl`: per-epoch sample counts, losses, F1 and measured times.

Each completed epoch writes weights through a temporary file and rename.
Nonfinite losses/gradients fail the run instead of overwriting good weights.
Hold-only stage-2 batches skip the optimizer, including weight decay. All
validation examples are retained, including the last partial batch.

The native AdamW optimizer operates on real-valued views of complex parameters,
matching Python's real/imaginary component-wise moments. Autograd reconstructs
complex views for the forward pass. Evaluation and training use the same soft
Markov transitions; dropout is disabled during validation.

## Direct executable

To prepare tensors without training:

```bash
python toric_markov_model/scripts/train_trading_cpp.py \
  --data /path/to/market.csv --checkpoint-dir prepared_cpp --prepare-only
toric_markov_model/cpp/build/toric_train \
  --input prepared_cpp/native/input.pt --output-dir prepared_cpp/native \
  --epochs 10 --batch-size 32 --threads 1 --device cpu --stage 0
```

The executable needs only LibTorch at runtime, not Python. Raw native weights
must be converted with `train.cpp_bridge.convert_checkpoint` before Python
backtesting; the normal launcher handles conversion automatically. It is not
a CSV parser, live-trading service or test-period evaluator.

The launcher exports only training and the chronological selection half of
validation. The remaining half is reserved for `scripts/calibrate_trading.py`.
Best checkpoint, early stopping and learning-rate reduction use minimized
stage-specific validation loss, not thresholded F1. Native/Python checkpoints
record the exact selection/calibration boundaries.

Interchange uses TorchScript solely as a named tensor container; the model is
not traced, scripted or frozen. This avoids fixed dropout modes, frozen shapes
and custom-autograd export problems in the training graph. TorchScript APIs
are deprecated in the installed PyTorch and emit warnings; this container
format is versioned as bundle 1 and must be revalidated on a PyTorch upgrade.
Only load locally prepared/trusted archives. Existing legacy format-1 model
checkpoints and historical filtered NPZ datasets remain unsupported.

## Verify numerical parity

After building, run from the repository root:

```bash
python -m unittest discover -s toric_markov_model/tests \
  -t toric_markov_model -p test_cpp.py -v
```

Native tests compare Python and C++ logits, auxiliary predictions, loss,
gradients and one clipped AdamW update in all three stages with dropout zero.
They also cover checkpoint conversion with dropout enabled, selection-partition
metrics, partial batches, hold-only stage 2 and invalid arguments.
If the executable is absent, these tests are explicitly skipped; they do not
silently count as native verification. Set `TORIC_CPP_BINARY` for another build.

## Measure speed

Run after preparing a bundle (or reuse a completed run's `native/` directory):

```bash
python toric_markov_model/scripts/benchmark_trading_cpp.py \
  --bundle-dir prepared_cpp/native --batch-size 32 --threads 1 --repeats 3
```

The benchmark starts both implementations from the same tensors and weights,
warms up one epoch and reports median train-plus-validation CPU time. CSV
preparation, executable startup and checkpoint writing are excluded from both
epoch measurements. Independent shuffling/dropout streams mean it is not a
model-quality comparison. Use dropout zero for less timing variation. CPU
thread count and batch size must be equal when comparing implementations.

C++ removes Python dispatch overhead but uses the same tensor kernels. It does
not guarantee acceleration on every workload, particularly when GPU kernels
dominate runtime. Measure with representative dimensions and history before
committing to long runs. Exported data lives in host RAM; GPU batches are
transferred synchronously, so this is not an optimized asynchronous CUDA pipeline.

Local synthetic smoke measurement (not a market benchmark): 565 training and
115 validation windows, sequence 32, dimension 32, 16 states, one recurrent
layer, dropout 0, batch 32 and one CPU thread. After one warmup epoch, three
measured epochs gave medians of 4.31 s for Python and 3.51 s for C++ (**1.23×**).
The benchmark utility prints individual timings and the full configuration;
repeat it on your real dataset before assuming the same gain.
