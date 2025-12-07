# ResNet-Based Image Classification Framework

![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

This repository contains a deep learning pipeline for image classification tasks using Transfer Learning with ResNet-18. It covers both binary classification (Cat vs. Dog) and multi-class problems with data imbalance (CIFAR-10).

## Project Structure

The codebase is organized into modular scripts:

- main.py: Core training engine for the binary classification task. Handles the training loop and validation.
- pretreat.py: Handles data augmentation, dataset splitting, and model initialization.
- predict.py: Inference script that processes the test dataset and generates a submission.csv.
- cifar_10_train.py: Experimental module for handling class imbalance on CIFAR-10 using weighted loss and sampling.

## Key Features

### 1. Transfer Learning
Used a ResNet-18 backbone pre-trained on ImageNet. I replaced the final fully connected layer to match the target classes and used an Adam optimizer with StepLR scheduler for stable convergence.

### 2. Data Augmentation
To improve generalization, I implemented a pipeline in pretreat.py including:
- RandomResizedCrop (224x224)
- RandomRotation and HorizontalFlip
- ColorJitter (brightness, contrast, etc.)
- Standard ImageNet normalization

### 3. Handling Data Imbalance
In the CIFAR-10 experiment (cifar_10_train.py), I simulated a long-tail distribution and tested two strategies:
- Class-Weighted Cross Entropy Loss
- WeightedRandomSampler (oversampling minority classes)

## Visualization

The training pipeline plots Loss and Accuracy curves in real-time. It also generates a prediction grid (Green = Correct, Red = Incorrect) to visualize model performance.

## Getting Started

### Prerequisites

pip install torch torchvision numpy pandas matplotlib imageio tqdm

### Usage

1. Train the Cat/Dog Classifier
Ensure dataset is in ./datasets_catdog/
python main.py

2. Run Inference
python predict.py

3. Run Imbalance Experiments (CIFAR-10)
python cifar_10_train.py

## Author

Yiqing Han
M.Sc. Student, NTU
