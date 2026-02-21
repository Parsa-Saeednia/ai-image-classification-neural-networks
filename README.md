# Image Classification with Fully Connected and Convolutional Neural Networks

## Course and Assignment Context

This repository contains **Assignment 4 (CA4)** of the **Artificial Intelligence** course  
(Computer Engineering) at the **University of Tehran**.

The assignment investigates image classification using:
- a **Fully Connected Neural Network**
- a **Convolutional Neural Network (CNN)**

implemented in **PyTorch**, and compares their performance on the **CIFAR‑10** dataset.

## Introduction

Neural networks can be designed with different architectural assumptions about the input data.

In this assignment, two fundamentally different architectures are examined:

1. **Fully Connected Neural Networks**, where images are flattened and treated as vectors
2. **Convolutional Neural Networks**, which explicitly exploit spatial structure in images

The goal is to understand how architectural choices affect learning behavior and classification
performance on image data.

## Problem Definition

The task of this assignment is to perform **image classification** using PyTorch.

You are required to:
1. Train a fully connected neural network on image data
2. Train a convolutional neural network on the same data
3. Compare the results of the two models under controlled conditions

To ensure fairness, both models are:
- trained on the same dataset
- evaluated using the same splits
- trained for the same number of epochs
- designed to have approximately the same number of trainable parameters


## Dataset: CIFAR‑10

The dataset used in this project is **CIFAR‑10**, a standard benchmark dataset in
machine learning and computer vision.

Dataset properties:
- 60,000 RGB images
- Image size: 32 × 32 pixels
- 10 classes
- 6,000 images per class

The CIFAR‑10 classes are:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Data Splitting

The dataset is split as follows:

- Training set: 45,000 images
- Validation set: 5,000 images
- Test set: 10,000 images

The training and validation sets are obtained by splitting the original training portion
of CIFAR‑10 using a random split.

## Data Loading and Normalization

Data loading is performed using PyTorch `Dataset` and `DataLoader`.

- Batch size: 512
- Shuffling is disabled to ensure reproducibility during feature‑space analysis

All images are normalized using channel‑wise statistics:
- Mean: (0.491, 0.482, 0.446)
- Standard deviation: (0.247, 0.243, 0.261)

## Dataset Visualization

To verify correct preprocessing and labeling, the dataset is visualized.

For each of the 10 classes:
- 5 random images are displayed
- Images are unnormalized before visualization

This step ensures correct data loading and provides qualitative insight into the dataset.

## Fully Connected Neural Network

### Motivation

In a fully connected neural network, images are flattened into vectors and fed directly
into dense layers.

This architecture does not explicitly use spatial information, but serves as a useful
baseline for comparison with convolutional networks.

### Architecture

Input:
- 3 × 32 × 32 image flattened to a 3072‑dimensional vector

Structure:
- Linear layers
- ReLU activations
- Dropout with probability 0.6
- Final classification layer with 10 outputs

### Trainable Parameter Calculation (Fully Connected Network)

The fully connected network is explicitly designed to have approximately
33.7 million trainable parameters.

Manual calculation:

(3072 × 4096 + 4096)  
+ (4096 × 4096 + 4096)  
+ (4096 × 10 + 10)

Total:
33,768,042 trainable parameters

The result is verified using `torchsummary`.

### Training Setup (Fully Connected Network)

- Loss function: CrossEntropyLoss
- Optimizer: Adam
- Learning rate: 0.0001
- Number of epochs: 30 (fixed)

## Convolutional Neural Network (CNN)

### Motivation

Convolutional Neural Networks are specifically designed for image data.

They leverage:
- local receptive fields
- parameter sharing
- hierarchical feature extraction

This architecture is expected to outperform fully connected networks on image tasks.

### CNN Architecture

The CNN consists of:
- multiple convolutional layers
- batch normalization layers
- non‑linear activations
- fully connected layers for classification

The architecture follows the structure provided in the assignment notebook.

### Trainable Parameter Calculation (CNN)

All trainable parameters are calculated manually.

#### Linear Layers

- Linear‑25: 22,121,100 parameters  
- Linear‑28: 2,765,824 parameters  
- Linear‑31: 10,250 parameters  

#### Total Trainable Parameters

The sum of all convolutional, batch normalization, and linear layers is:

33,759,126 trainable parameters

The result is verified using `torchsummary.summary`.

### Training Setup (CNN)

- Loss function: CrossEntropyLoss
- Optimizer: Adam
- Learning rate: 0.0001
- Number of epochs: 30 (fixed)

## Training Procedure

For each epoch, the following metrics are recorded:
- Training loss
- Training accuracy
- Validation loss
- Validation accuracy

All metrics are stored for later visualization and analysis.

## Model Saving

After training, the CNN model is saved to disk to avoid retraining:

- cnn.pth

The saved model can be reloaded later for evaluation and analysis.

## Evaluation

The trained model is evaluated on the test set.

Reported metrics:
- Test loss
- Test accuracy

The target accuracy specified in the assignment is **above 80%**.

## Visualization of Incorrect Predictions

24 randomly selected test images that are **incorrectly classified** are visualized.

Images are unnormalized before display, and for each image:
- predicted label
- true label

are shown to enable qualitative error analysis.

## Feature Space Exploration

### Feature Space Extraction

For each training sample, the learned feature representation is extracted from the model.

The resulting feature tensor has shape:
(45,000, N)

These features are saved for later reuse.

### K‑Nearest Neighbors in Feature Space

To analyze the learned representation:

1. Correctly classified test samples are selected
2. Feature vectors are extracted for these samples
3. Cosine distance is used to find the 5 nearest neighbors in the training feature space
4. Nearest neighbors are visualized

This analysis helps reveal semantic similarity in the learned representation.

### t‑SNE Visualization

To further explore the feature space:

- 2,000 random training samples are selected
- Feature vectors are reduced to 2 dimensions using t‑SNE
- Points are visualized in a 2D plane
- Colors correspond to class labels

This visualization provides insight into class separability.

## Feature Map Visualization

Intermediate feature maps of the CNN are visualized.

The model is clipped at an intermediate convolutional layer, and the output feature maps
are plotted to observe what patterns the filters respond to.

## Repository Structure

├── AI_S04_CA4.ipynb   # Complete implementation and analysis (1997 lines)
├── README.md
└── .gitignore

## How to Run

1. Install Python 3.x
2. Install dependencies:
   pip install torch torchvision numpy matplotlib scikit-learn
3. Open and run `AI_S04_CA4.ipynb` sequentially

## Academic Nature

This project is educational and experimental.

Its purpose is to:
- compare neural network architectures
- understand representation learning
- analyze model behavior

It is not intended for production deployment.

## Authorship

Author: Parsa Saeednia  
Course: Artificial Intelligence  
Institution: University of Tehran

## License

This repository is released under the MIT License.

The license applies only to original code and documentation.
The CIFAR‑10 dataset and assignment description are used strictly for educational purposes.
