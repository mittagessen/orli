**conformer ocr**
========

conformer_ocr is a drop-in replacement for kraken's CNN-LSTM line text
recognizer based on a slightly modified conformer architecture. 

Installation
------------

::

        $ pip install .


Training
--------

Options are largely identical to those offered by `ketos test`, including
possible data set formats.

::

        $ cocr -d cuda train -f binary --workers 32 *.arrow

Default hyperparameters are optimized for large datasets (~50k lines), trained
with reasonably large batch sizes (32) on a GPU with at least 16Gb memory. 

Inference
---------

Inference is supported with:

::

        cocr ocr -i input output -m model.ckpt ...

Inputs can be defined as with kraken's inference tools. A segmentation must be
provided from XML files in ALTO or Page XML format. Outputs in any of kraken's
formats are supported.

