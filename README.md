# Orli

Orli (**o**rdered **r**egression of **li**nes) is a layout analysis method
performing the text line detection and reading order determination subtasks
jointly.

Orli consists of a ConvNeXtV2 vision encoder (or optionally a RegNetX-8GF
encoder via the `regnetx` model variant), an adapter module projecting
multi-scale feature maps into a shared embedding space, and a transformer
decoder with cross-attention. The autoregressive decoder predicts baselines by
regressing local-frame baseline vectors through iterative refinement. Training
targets are fixed-size, arc-length-resampled baseline polylines; the regressed
vector stores the baseline center, length, orientation, and normal offsets for
the sampled points.

## Model Variants

| Variant | Encoder | Params | Notes |
|---|---|---|---|
| `pico` | ConvNeXtV2-pico | ~28 M | Fastest, 2-level neck |
| `tiny` | ConvNeXtV2-tiny | ~28 M | Default |
| `small` | ConvNeXtV2-small | ~50 M | Higher capacity |
| `regnetx` | RegNetX-8GF | ~39 M | timm `regnetx_080.tv2_in1k` weights; same neck topology as `tiny` |

Select a variant with `--model-variant` or set `model_variant:` in the experiment YAML.

## Installation

```bash
$ pip install .
```

## Dataset Preparation

Orli needs to be trained on datasets precompiled from PageXML or ALTO files.
The compiler stores each line as a normalized baseline `polyline` in source-file
order; the training loader resamples those polylines for the configured baseline
point count.

```bash
$ orli compile -o dataset.arrow --allow-textless *.xml
```

If you have recent GPUs or the input images are very large, the training is
probably I/O-bound. In that case it can help to resize the images in the
dataset to the input size of the network. For the default (1280, 960):

```bash
$ orli compile -o dataset.arrow --allow-textless -r 1280 960 *.xml
```

The compilation **always** uses the implicit reading order, i.e., the sequence
of line elements in the source files. If other reading orders are defined they
will be ignored.

## Training and Fine-tuning

Training can be configured using the command line or experiment YAML files (preferred):

```yaml
precision: bf16-mixed
device: auto
num_workers: 12
num_threads: 1
train:
  training_data:
    - orli_train.lst
  evaluation_data:
    - orli_val.lst
  checkpoint_path: experiments/base_orli
  image_size: [1280, 960]
  optimizer: AdamW
  epochs: 16
  lrate: 1e-4
  weight_decay: 1e-4
  schedule: cosine
  cos_t_max: 16
  cos_min_lr: 1e-5
  warmup: 2000
  augment: true
  batch_size: 8
  val_batch_size: 16
  accumulate_grad_batches: 8
  baseline_num_points: 16
```

Train the model:

```bash
$ orli --config experiment.yaml train
```

Resume training from a checkpoint:

```bash
$ orli --config experiment.yaml train --resume /path/to/checkpoint.ckpt
```

Fine-tune from an existing model:

```bash
$ orli --config experiment.yaml train --load /path/to/model.safetensors
```

### Checkpoint Conversion

Checkpoints need to be converted into safetensors format before being usable
for inference and testing:

```bash
$ ketos convert -o model.safetensors checkpoint.ckpt
```

## Inference

Inference is implemented through the plugin system in kraken (>= 7):

```bash
$ kraken -i input.jpg output.xml -a segment -bl -i model.safetensors
```

## Testing

Evaluate a model on an arrow test dataset, computing baseline detection metrics
inspired by the
[TranskribusEvaluationScheme](https://github.com/Transkribus/TranskribusBaseLineEvaluationScheme)
and reading order metrics (Spearman footrule, Kendall tau):

```bash
$ orli test --load model.safetensors test.arrow
```
