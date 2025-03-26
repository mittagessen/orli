# Orli

Orli (**o**rdered **r**egression of **li**nes) is a layout analysis method
performing the text line detection and reading order determination subtasks.

Orli consists of a Swin vision transformer encoder (shared with
[party](https://github.com/mittagessen/party)) and a small transformer decoder. 
The autoregressive decoder predicts baselines by regressing the normalized
coordinates of the control points of a cubic Bézier curve.

## Installation

  $ pip install .

## Fine Tuning

Orli needs to be trained on datasets precompiled from PageXML or ALTO files
containing baseline information for each line. The binary dataset format is
**NOT** compatible with kraken but is shared with
[party](https://github.com/mittagessen/party). Please install that tools and compile with:

  $ party compile -o dataset.arrow *.xml

## Checkpoint conversion

Checkpoints need to be converted into a safetensors format before being usable for inference and testing.

  $ orli convert -o model.safetensors checkpoint.checkpoint

## Inference

TO BE IMPLEMENTED

## Testing

## Performance

Training and inference resource consumption is highly dependent on various
optimizations being enabled. Torch compilation which is required for various
attention optimizations is enabled per default but lower precision training
which isn't supported on CPU needs to be configured manually with `party
--precision bf16-mixed ...`.
