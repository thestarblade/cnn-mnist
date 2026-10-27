# cnn-mnist

A small convolutional neural network in pure C that classifies handwritten
MNIST digits. No dependencies beyond the C standard library, no BLAS, no
external build system.

Based on [tinymnist](https://github.com/itskonrad/tinymnist), which started as a
784-256-10 multilayer perceptron. The MLP baseline is preserved as the
`baseline-mlp` tag.

## Architecture

```
input        1 x 28 x 28
conv 3x3     8 filters, padding 1, ReLU     ->   8 x 28 x 28
max pool     2 x 2                          ->   8 x 14 x 14
conv 3x3    16 filters, padding 1, ReLU     ->  16 x 14 x 14
max pool     2 x 2                          ->  16 x 7 x 7
flatten                                     ->         784
dense                                       ->          64, ReLU
dense                                       ->          10, softmax
```

Cross-entropy loss, SGD with momentum, He initialisation, batch size 32,
learning rate 0.01. The 60000 provided training images are split 80/20 into a
training and a held-out set.

## Quick start

```bash
make
./build/release/cnn
```

## Results

`./build/release/cnn --seed 2026` on the held-out 12000 images:

| Epoch | Accuracy | Avg loss | Time |
|-------|----------|----------|------|
| 1     | 97.80%   | 0.2510   | 10.23 s |
| 2     | 98.08%   | 0.0693   | 10.69 s |
| 5     | 98.78%   | 0.0328   | 10.47 s |
| 10    | **98.93%** | 0.0142 | 9.86 s |

About 11 seconds per epoch on one core, so a full run takes under two minutes.

## Command line

```
Usage: ./build/release/cnn [options]

Options:
  --epochs N        number of epochs (default 10, 0 = evaluate only)
  --batch-size N    samples per gradient step (default 32)
  --lr F            learning rate (default 0.0100)
  --momentum F      SGD momentum (default 0.90)
  --seed N          random seed (default: current time)
  --data-dir DIR    directory holding the IDX files (default data)
  --load PATH       start from a saved checkpoint
  --save PATH       write the trained weights to PATH
  -h, --help        show this help
```

Checkpoints store every weight and bias together with the layer dimensions, so
a file from a different architecture is rejected instead of loaded blindly:

```bash
./build/release/cnn --epochs 5 --save model.cnn
./build/release/cnn --epochs 0 --load model.cnn
```

The train/held-out split comes from the seeded shuffle, so pass the same
`--seed` when comparing two runs; otherwise the accuracy is measured on a
different subset.

## Development

```bash
make              # optimized build   -> build/release/cnn
make debug        # ASan + UBSan      -> build/debug/cnn
make test         # unit tests, also under ASan + UBSan
make EPOCHS=1     # quick training smoke test
make clean
```

| Path | Contents |
|------|----------|
| `main.c` | CLI, training loop, evaluation |
| `mnist.c/h` | IDX loader and shuffle |
| `layer.c/h` | dense layer, softmax, ReLU backward |
| `conv.c/h` | conv/pool forward and backward, im2col |
| `cnn.c/h` | network definition, forward, backward, SGD step |
| `checkpoint.c/h` | weight save/load |
| `tests/` | unit and finite-difference gradient tests |

Convolution goes through `im2col`, which turned an epoch from 79.2 s into
11.6 s. The direct implementations are kept as `conv2d_direct_*` so the tests
can check that the optimised path produces identical outputs and gradients.

## License

[MIT](LICENSE), see the original [tinymnist](https://github.com/itskonrad/tinymnist).
