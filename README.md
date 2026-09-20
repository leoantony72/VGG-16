# VGG16 on CIFAR-10

A VGG16 convolutional neural network built from scratch in PyTorch, trained on the CIFAR-10 dataset.

## What is VGGNet?

VGGNet is a deep CNN architecture that uses small 3x3 convolution filters stacked repeatedly, instead of larger filters. Stacking small filters gives the same receptive field as a larger filter but with fewer parameters and more non-linearity.

## Architecture

The network has 5 convolutional blocks followed by a classifier.

| Block | Layers | Output channels |
|---|---|---|
| Block 1 | 2x Conv3x3 + MaxPool | 64 |
| Block 2 | 2x Conv3x3 + MaxPool | 128 |
| Block 3 | 3x Conv3x3 + MaxPool | 256 |
| Block 4 | 3x Conv3x3 + MaxPool | 512 |
| Block 5 | 3x Conv3x3 + MaxPool | 512 |

Every conv layer uses `kernel_size=3, stride=1, padding=1`.
Every MaxPool uses `kernel_size=2, stride=2`.

Classifier: Flatten -> Linear(4096) -> ReLU -> Dropout -> Linear(4096) -> ReLU -> Dropout -> Linear(num_classes)

## Dataset

- **CIFAR-10**: 60,000 images, 32x32 pixels, 3 color channels, 10 classes
- Since images are 32x32 (not 224x224 like ImageNet), the feature map after block 5 is 512x1x1 instead of 512x7x7

## Key training components

| Component | Value | Why |
|---|---|---|
| BatchNorm2d | after every Conv2d | stabilizes training in deep networks |
| Optimizer | SGD | standard choice for CNN training |
| momentum | 0.9 | speeds up and smooths convergence |
| weight_decay | 5e-4 | reduces overfitting by penalizing large weights |
| learning rate | 0.01 | balances speed and stability |

## How to run

```bash
python vgg16.py
```

This will download CIFAR-10, train the model, and print train/test loss and accuracy per epoch.

## Expected results

- Without BatchNorm/momentum: model fails to learn (stuck at ~10% accuracy)
- With BatchNorm + proper optimizer, ~20 epochs: ~75-85% accuracy
- With more epochs (50-100+) and a learning rate scheduler: ~90%+ accuracy
