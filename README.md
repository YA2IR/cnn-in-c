# Convolutional Neural Network (CNN) in C
| ![Image 1](images/img1.png) | ![Image 2](images/img2.png) | ![Image 3](images/img3.png) | ![Image 4](images/img4.png) |
|---------------------------|---------------------------|---------------------------|---------------------------|

*Example feature maps visualized - just for fun.*
## Overview
This is a Convolutional Neural Network (CNN) implementation in C!

The project includes a full CNN with most expected features:
- Includes all the necessary layers: 
    - Convolution 
    - Maxpooling 
    - Flattening
    - and Fully-Connected
- Activations, support for batch-oriented training, and gradient clipping.
- Matrix operations, customizable hyperparameters, etc.

It should work out-of-the-box since the data is intentionally left in the repo (in the `mnist_dataset` folder).
## Motivation & Background
This started as a simple Artificial Neural Network project to explore neural networks & learn how they work at a high level, and I didn't even know CNNs existed. After finishing it, I encountered CNNs and was mind blown by their [spatial invariance](https://medium.com/@Orca_Thunder/the-secret-sauce-invariance-of-cnns-f7042f4457f3) among other things, so I extended it and here we are!

## Limitations
The implementation still lacks some typical features, like padding, Adam, and batch normalization.

This is a work in progress, and is going through refactoring, so expect slight anomalies (e.g. my weight initializations are sensitive to the random seed). And also, I plan to use a normal tensor datatype (1D arrays rather than multidimensional ones)
## Usage

### Compilation & Running
After cloning this repo, it should be straightforward to compile and run (clang and gcc):
```
clang -O3 -o main main.c matrix.c cnn.c
./main
```
Or see/use run.sh for explicit OpenMP support

You can adjust the hyperparameters in `hyperparams.h`, but the architecture is in `main.c`

### Example output:

```
Mac:~ clang -O3 -o main main.c matrix.c cnn.c && ./main

------------------------
Epoch 1: train_acc = 85.2%, val_acc = 91.6%
Epoch 2: train_acc = 90.6%, val_acc = 93.7%
Epoch 3: train_acc = 91.7%, val_acc = 94.5%
Epoch 4: train_acc = 92.2%, val_acc = 94.7%
Epoch 5: train_acc = 92.6%, val_acc = 94.7%

------------------------
test accuracy: 93.5%

Exit success.
```

## Credits
- The mnist loader `mnist.h` is taken from https://github.com/projectgalateia/mnist
- MNIST data are taken from [Yann LeCun](http://yann.lecun.com/) et al 