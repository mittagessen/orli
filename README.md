# Orli

Orli (**o**rdered **r**egression of **li**nes) is a layout analysis method
performing the text line detection and reading order determination subtasks
jointly.

Orli consists of a ConvNeXtV2-tiny vision encoder, an adapter module projecting
multi-scale feature maps into a shared embedding space, and a transformer
decoder with cross-attention. The autoregressive decoder predicts baselines by
regressing the normalized control points of cubic Bezier curves through
iterative refinement.

## Installation

```bash
$ pip install .
```

## Dataset Preparation

Orli needs to be trained on datasets precompiled from PageXML or ALTO files
containing baseline information for each line in correct reading order. The
binary dataset format is shared with
[party](https://github.com/mittagessen/party). Install party and compile with:

```bash
$ party compile -o dataset.arrow --allow-textless *.xml
```

If you have recent GPUs or the input images are very large, the training is
probably I/O-bound. In that case it can help to resize the images in the
dataset to the input size of the network. For the default (1280, 960):

```bash
$ party compile -o dataset.arrow --allow-textless -r 1280 960
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
$ kraken -i input.jpg output.xml -a segment -bl -m model.safetensors
```

## Testing

Evaluate a model on an arrow test dataset, computing baseline detection metrics
inspired by the
[TranskribusEvaluationScheme](https://github.com/Transkribus/TranskribusBaseLineEvaluationScheme)
and reading order metrics (Spearman footrule, Kendall tau):

```bash
$ orli test --load model.safetensors test.arrow
```
