# Orli + RegNetX-8GF — Quickstart

Training orli with the `regnetx` backbone variant instead of the default
ConvNeXtV2.  Everything else — the autoregressive decoder, curve regression
head, anchor system, data pipeline, and inference API — is identical to the
original project.

---

## What changed

Three files were edited, nothing was removed:

| File | Change |
|---|---|
| `orli/orli.py` | Added `_RegNetXFeatureInfo`, `_RegNetX8GFEncoder` wrapper classes and a one-line branch in `OrliModel.__init__` to use them when `encoder_name == 'regnetx_8gf'` |
| `orli/configs.py` | Added `'regnetx'` entry to `MODEL_VARIANTS` |
| `orli/cli/train.py` | Added `'regnetx'` to `--model-variant` choices |
| `pyproject.toml` | Added `torchvision>=0.20.0` dependency |

Drop those four files into the orli source tree and reinstall.

---

## Install

```bash
# Clone orli (feature/better_curve_representation branch)
git clone -b feature/better_curve_representation https://github.com/mittagessen/orli.git
cd orli

# Drop in the four edited files:
#   orli/orli.py         → orli/orli.py
#   orli/configs.py      → orli/configs.py
#   orli/cli/train.py    → orli/cli/train.py
#   pyproject.toml       → pyproject.toml

pip install -e ".[dev]"
# torchvision is now a declared dependency and will be installed automatically.
# The RegNetX-8GF ImageNet weights (~200 MB) are downloaded on first run.
```

---

## Compile training data

orli requires kraken's binary dataset format (`.arrow`).  Use `ketos compile`
from the kraken package to compile your PAGE-XML or ALTO ground-truth into
binary shards.

```bash
# Compile a directory of PAGE-XML files into a binary shard:
ketos compile \
    -f page \
    -o dataset/train.arrow \
    /path/to/pagexml/train/

ketos compile \
    -f page \
    -o dataset/val.arrow \
    /path/to/pagexml/val/
```

For large datasets compile with `--workers N` to parallelise image loading:

```bash
ketos compile -f page --workers 16 -o dataset/train.arrow /path/to/pagexml/
```

---

## Training from scratch

```bash
orli train \
    --model-variant regnetx \
    -t dataset/train.arrow \
    -e dataset/val.arrow \
    -o checkpoints/regnetx_run1 \
    -N 32 \
    --quit fixed \
    -B 4 \
    --accumulate-grad-batches 8 \
    --lrate 1e-4 \
    --schedule cosine \
    --cos-max 32 \
    --cos-min-lr 1e-5 \
    --warmup 2000 \
    --weight-decay 1e-4 \
    --gradient-clip-val 1.0 \
    --augment \
    --image-size 1280 960
```

**Effective batch size** = `batch_size × accumulate_grad_batches` = 4 × 8 = 32.
On an RTX 3090 (24 GB) `batch_size=4` with `bf16-mixed` precision is comfortable;
reduce to 2 if you hit OOM.

---

## Key options explained

| Option | Default | Notes |
|---|---|---|
| `--model-variant regnetx` | `tiny` | Selects the RegNetX-8GF backbone |
| `-B` / `--batch-size` | 8 | Per-GPU batch size |
| `--accumulate-grad-batches` | 8 | Multiply with batch size for effective batch |
| `--image-size H W` | 1280 960 | Network input resolution — keep aspect ratio of your pages |
| `--lrate` | 1e-4 | Peak learning rate (full LR for neck/decoder, 0.1× for encoder) |
| `--warmup` | 2000 | Linear LR warmup steps |
| `--augment` | off | Enable albumentations augmentation (recommended) |
| `--quit fixed` / `--quit early` | fixed | `early` uses early stopping monitored on `val_metric` |
| `--lag N` | — | Early-stopping patience (validation rounds) |
| `--baseline-num-points N` | auto | Number of fixed arc-length points per baseline curve |
| `--direct-point-regression` | off | Regress points directly instead of local-frame parameters |

---

## Encoder learning rate

`configure_optimizers` applies a 0.1× LR multiplier to the encoder
(`encoder_lr = lrate * 0.1`) — the same as for the ConvNeXtV2 variants.
RegNetX-8GF has supervised ImageNet weights rather than FCMAE pretraining, so
it may adapt slightly faster. If you see slow early loss reduction you can
override this by patching `configure_optimizers` in `orli/model.py`:

```python
# In OrliSegmentationModel.configure_optimizers():
encoder_lr = self.hparams.config.lrate * 0.2   # raise from 0.1 to 0.2
```

---

## Resuming an interrupted run

```bash
orli train \
    --model-variant regnetx \
    --resume checkpoints/regnetx_run1/checkpoint_08-0.1234.ckpt \
    -t dataset/train.arrow \
    -e dataset/val.arrow \
    -o checkpoints/regnetx_run1
```

---

## Warm-starting from a checkpoint

To continue training from an existing weights file (e.g. after converting a
checkpoint to safetensors with `ketos convert`):

```bash
orli train \
    --model-variant regnetx \
    -i checkpoints/regnetx_run1/best_0.1234.safetensors \
    -t dataset/train2.arrow \
    -e dataset/val2.arrow \
    -o checkpoints/regnetx_finetune
```

---

## Monitoring

```bash
# TensorBoard
orli train --logger tensorboard ...
tensorboard --logdir checkpoints/regnetx_run1

# Weights & Biases
orli train --logger wandb ...
```

---

## Recommended recipe for a multilingual manuscript corpus

```bash
# Phase 1: 32 epochs, frozen encoder (adapt neck + decoder first)
orli train \
    --model-variant regnetx \
    -t dataset/train.arrow \
    -e dataset/val.arrow \
    -o checkpoints/phase1 \
    -N 32 --quit fixed \
    -B 4 --accumulate-grad-batches 8 \
    --lrate 1e-4 --schedule cosine --cos-max 32 --cos-min-lr 1e-5 \
    --warmup 2000 --weight-decay 1e-4 \
    --gradient-clip-val 1.0 --augment \
    --image-size 1280 960

# Phase 2: finetune all parameters from best phase-1 checkpoint
orli train \
    --model-variant regnetx \
    -i checkpoints/phase1/best_0.XXXX.safetensors \
    -t dataset/train.arrow \
    -e dataset/val.arrow \
    -o checkpoints/phase2 \
    -N 16 --quit fixed \
    -B 4 --accumulate-grad-batches 8 \
    --lrate 3e-5 --schedule cosine --cos-max 16 --cos-min-lr 3e-6 \
    --warmup 500 --weight-decay 1e-4 \
    --gradient-clip-val 1.0 --augment \
    --image-size 1280 960
```

Phase 1 is equivalent to a frozen-encoder run (the 0.1× encoder LR means the
encoder moves very little). Phase 2 unlocks full fine-tuning at a lower peak LR.

---

## Architecture summary

| Component | RegNetX variant | ConvNeXtV2-tiny (default) |
|---|---|---|
| Backbone | RegNetX-8GF (torchvision, 39 M params) | ConvNeXtV2-tiny (timm, 28 M params) |
| Pretrain | ImageNet-1K V2 supervised | FCMAE self-supervised |
| Selected stages | C3/C4/C5 (strides 8/16/32) | stages 1/2/3 (strides 8/16/32) |
| Stage channels | 240 / 720 / 1920 | 192 / 384 / 768 |
| Neck input_proj | 1×1 Conv per scale (absorbs channel diff) | same |
| Neck hidden_dim | 256 | 256 |
| Spatial sizes (1280×960) | 160×120 / 80×60 / 40×30 | identical |
| Decoder | 12-layer Llama-3.2 cross-attention (576-dim) | identical |
| Regressor | iterative anchor refinement | identical |

The spatial token counts fed to the decoder cross-attention are **identical**
between `regnetx` and `tiny` at any given image resolution, so the two variants
are architecturally interchangeable in the decoder and regressor.
