# Orli

Orli (**o**rdered **r**egression of **li**nes) is a layout-analysis model that
detects text-line baselines and emits them directly in reading order. It is
designed for historical-document OCR pipelines that need explicit line geometry
without a separate reading-order heuristic.

The method is described in the article
[End-to-End Text Line Detection and Ordering](https://arxiv.org/abs/2606.04166).

## Installation

```bash
pip install .
```

Orli integrates with kraken 7 through its model plugin system.

## Model

The release base model is trained on 200000 pages spanning ten writing systems.
It is published available through HTRMoPo with DOI
[10.5281/zenodo.20558179](https://doi.org/10.5281/zenodo.20558179).

Download it with kraken:

```bash
kraken get 10.5281/zenodo.20558179
```

The command prints the model directory and the downloaded model file, necessary
for fine-tuning and programmatic inference.

Run baseline segmentation with kraken. This example writes PAGE XML:

```bash
kraken --precision bf16-mixed -i input.jpg output.xml -x segment -bl --model orli_base.safetensors
```

The base model *only* works in bfloat16 precision! Other precisions are likely
to cause runaway generation.

Programmatic inference uses the complete model path printed after download:

```python
from PIL import Image
from orli.pred import segment

im = Image.open("input.jpg")
segmentation = segment(im, "/path/to/kraken/download/orli_base.safetensors")
```

## Scores

Line metrics are computed using the cBAD evaluation score implemented in `orli
test`. Footrule is normalized Spearman footrule, where lower is better.

### Test Set

| Model | Precision | Recall | F1 | Cov. | Footrule | Kendall tau |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.9554 | 0.9564 | 0.9559 | 0.9667 | 0.0304 | 0.9649 |

### cBAD 2019

| Model | Precision | Recall | F1 | Cov. | Footrule | Kendall tau |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.9378 | 0.9302 | 0.9340 | 0.9406 | 0.0768 | 0.9113 |
| fine-tuned | 0.9395 | 0.9306 | 0.9351 | 0.9421 | 0.0720 | 0.9165 |

### Reading-Order Benchmarks

| Dataset | Model | Precision | Recall | F1 | Cov. | Footrule | Kendall tau |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OHG | base | 0.9940 | 0.9937 | 0.9938 | 0.9993 | 0.0033 | 0.9967 |
| FCR | base | 0.9894 | 0.9874 | 0.9884 | 0.9905 | 0.0028 | 0.9971 |
| ABP | base | 0.8505 | 0.7919 | 0.8201 | 0.8071 | 0.5372 | 0.2878 |
| ABP | fine-tuned | 0.8498 | 0.7806 | 0.8137 | 0.7931 | 0.0898 | 0.8972 |

## Dataset Preparation

Orli trains on Arrow datasets compiled from PageXML or ALTO files. The compiler
stores each line as a normalized baseline polyline in source-file order. The
arrow files are *NOT* compatible with kraken's compiled datasets.

```bash
orli compile -o dataset.arrow --allow-textless *.xml
```

For large images, pre-resizing during compilation reduces training I/O. The
base model uses a high-resolution input size of 1920x1440:

```bash
orli compile -o dataset.arrow --allow-textless -r 1920 1440 *.xml
```

Compilation uses the implicit reading order, i.e. the sequence of line elements
in the source file. Other reading-order annotations are ignored.

## Training and Fine-Tuning

Training and fine-tuning are configured either through command-line options or a
YAML file. For fine-tuning the released base model, keep the high-resolution
input size and load the downloaded `orli_base.safetensors` file:

```yaml
precision: bf16-mixed
device: auto
num_workers: 12
num_threads: 1
train:
  training_data:
    - train.arrow
  evaluation_data:
    - val.arrow
  checkpoint_path: experiments/orli_finetuned
  image_size: [1920, 1440]
  optimizer: AdamW
  epochs: 8
  lrate: 5e-5
  weight_decay: 1e-4
  schedule: cosine
  cos_t_max: 8
  cos_min_lr: 1e-5
  warmup: 1000
  augment: true
  batch_size: 8
  val_batch_size: 8
  accumulate_grad_batches: 8
  baseline_num_points: 16
```

```bash
orli --config finetune.yaml train --load "$MODEL"
orli --config finetune.yaml train --resume /path/to/checkpoint.ckpt
```

The training command writes the best checkpoint and converts it to safetensors
automatically. The resulting `best_*.safetensors` file can be used with
`kraken segment` in the same way as the base model.

## Evaluation

Evaluate a model on an Arrow dataset with baseline detection metrics and
reading-order metrics:

```bash
orli test --load model.safetensors test.arrow
```

## Citation

```bibtex
@misc{kiessling2026orli,
  title = {End-to-End Text Line Detection and Ordering},
  author = {Benjamin Kiessling},
  year = {2026},
  eprint = {2606.04166},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url = {https://arxiv.org/abs/2606.04166}
}
```

## License

Orli is released under the Apache License 2.0.
