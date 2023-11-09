# Improving Inference for Neural Image Compression

![Example SGA optimization landscape](results/sga_landscape.png)

Reimplementation in tensorflow 2.10.
Currently only the standalone SGA procedure (without bits-back) is supported.
See `demo.ipynb` for an example.

**UPDATE [2023]**: Please see the [shallow NTC repo](https://github.com/mandt-lab/shallow-ntc/) for the best and most up-to-date implementation of SGA and training/evaluation pipeline for neural image compression in tf2.

## Requirements
The code was tested on [tfc version 2.10.0](https://github.com/tensorflow/compression/releases/tag/v2.10.0).

All the dependencies can be installed by

    pip install tensorflow-compression==2.10.0


