# Improving Stacked Hourglass Networks with Regularization

Based on **Stacked Hourglass Networks for Human Pose Estimation.** [Alejandro Newell](http://www-personal.umich.edu/~alnewell/), Kaiyu Yang, and [Jia Deng](https://www.cs.princeton.edu/~jiadeng/). *European Conference on Computer Vision (ECCV)*, 2016. [Github](https://github.com/princeton-vl/pose-hg-train)

PyTorch code extended from [Github](https://github.com/princeton-vl/pytorch_stacked-hourglass). Implemented under advisors Alejandro Newell and Prof. Jia Deng.

## Overview

Newell et al. originally reported 0.881 validation accuracy using 8HG model on MPII. [Here](https://github.com/princeton-vl/pytorch_stacked-hourglass) we get validation accuracy of 0.885 using a 2HG model and 0.901 using an 8HG model. In this implementation, validation accuracies of 0.887 and 0.906 are achieved by adding mean-normalization, cutout, and vertical flipping. Test number of 0.913 is also achieved, as opposed to authors' 0.909.

## Getting Started

This repository provides everything necessary to train and evaluate a single-person pose estimation model on MPII. If you plan on training your own model from scratch, we highly recommend using multiple GPUs.

Requirements:

- Python 3 (code has been tested on Python 3.6)
- PyTorch (code tested with 0.4)
- CUDA and cuDNN
- Python packages (not exhaustive): opencv-python, tqdm, cffi, h5py, scipy (tested with 1.1.0)

Structure:
- ```data/```: data loading and data augmentation code
- ```models/```: network architecture definitions
- ```task/```: task-specific functions and training configuration
- ```utils/```: image processing code and miscellaneous helper functions
- ```train.py```: code for model training
- ```test.py```: code for model evaluation

#### Dataset
Download the full [MPII Human Pose dataset](http://human-pose.mpi-inf.mpg.de/), and place the images directory in data/MPII/

#### Training and Testing

To train a network, call:

```python train.py -e test_run_001``` (```-e,--exp``` allows you to specify an experiment name)

To continue an experiment where it left off, you can call:

```python train.py -c test_run_001```

All training hyperparameters are defined in ```task/pose.py```, and you can modify ```__config__``` to test different options. It is likely you will have to change the batchsize to accommodate the number of GPUs you have available.

Once a model has been trained, you can evaluate it with:

```python test.py -c test_run_001```

The option "-m n" will automatically stop training after n total iterations (if continuing, would look at total iterations)

#### Pretrained Models

An 8HG pretrained model is available [here](http://www-personal.umich.edu/~cnris/regularization_2hg/checkpoint.pth.tar). It should yield validation accuracy of 0.906.

A 2HG pretrained model is available [here](http://www-personal.umich.edu/~cnris/regularization_8hg/checkpoint.pth.tar). It should yield validation accuracy of 0.887.

Models should be formatted as exp/<exp_name>/checkpoint.pth.tar

#### Training/Validation split

The train/val split is same as that found in authors' [implementation](https://github.com/princeton-vl/pose-hg-train)
