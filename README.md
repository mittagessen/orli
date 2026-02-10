# Orli

Orli (**o**rdered **r**egression of **li**nes) is a layout analysis method
performing the text line detection and reading order determination subtasks.

Orli consists of a Swin vision transformer encoder (shared with
[party](https://github.com/mittagessen/party)) and a small transformer decoder. 
The autoregressive decoder predicts baselines by regressing the normalized
coordinates of the control points of a cubic Bézier curve.

## Installation

```bash
  $ pip install .
```

## Dataset preparation

Orli needs to be trained on datasets precompiled from PageXML or ALTO files
containing baseline information for each line in correct reading order. The
binary dataset format is **NOT** compatible with kraken but is shared with
[party](https://github.com/mittagessen/party). Please install party and compile
with:

```bash
$ party compile -o dataset.arrow --allow-textless *.xml
```

If you've got recent GPUs or the input images are very large, the training is
probably I/O-bound. In that case it can help to resize the images in the
dataset to the input size of the network. For the default (1280, 960):

```bash
    $ party compile -o dataset.arrow --allow-textless -r 1280 960
```

The compilation **always** uses the implicit reading order, i.e., the sequence
of line elements in the source files. If other reading orders are defined they
will be ignored. 

## Training and Fine Tuning

Training can be configured using the command line or experiment YAML files (preferred):

```yaml
    precision: bf16-mixed
    device: auto 
    num_workers: 12
    num_threads: 1
    train:
      # training data manifests
      training_data:
         - orli_train.lst
      evaluation_data:
        - orli_val.lst
      # directory to save checkpoints in
      checkpoint_path: experiments/base_orli
      image_size: [1280, 960]
      # base configuration of training epochs and LR schedule
      optimizer: AdamW
      epochs: 24
      lrate: 3e-4
      schedule: cosine
      cos_t_max: 24
      cos_min_lr: 1e-5
      warmup: 2000
      augment: true
      batch_size: 12
      val_batch_size: 16
      accumulate_grad_batches: 8
```

then, train the model:

```bash
$ orli --config experiment.yaml train
```

You can resume training with:

```bash
$ orli --config experiment.yaml train --resume /path/to/checkpoint.ckpt
```

and fine-tune from an existing model:

```bash
$ orli --config experiment.yaml train --load /path/to/model.safetensors
```

### Checkpoint conversion

Checkpoints need to be converted into a safetensors format before being usable for inference and testing.

```bash
  $ kraken convert -o model.safetensors checkpoint.ckpt
```

## Inference

Inference is implemented through the plugin system in the kraken 7 upcoming release:

```bash
$ kraken -i input.jpg output.xml -a segment -bl -m model.safetensors
```

## Testing

```bash
$ orli test --load model.safetensors test.arrow
```
