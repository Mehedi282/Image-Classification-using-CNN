# CIFAR-10 Image Classification using CNN

## Overview

This project implements a **Convolutional Neural Network (CNN)** using **TensorFlow** and **Keras** to classify images from the **CIFAR-10** dataset. The model is trained on CIFAR-10 and can predict the class of a given image uploaded by the user.


[Open the Colab Project](https://colab.research.google.com/drive/19prG-O72Ukc49zjEYVmvfM39797IV1dV#scrollTo=2XiDbUACOSVV)


## Features

- Trains a CNN model on the CIFAR-10 dataset.
- Saves and loads the trained model (`cifar10_cnn_model.h5`).
- Allows users to upload an image for prediction.
- Uses **OpenCV** for image preprocessing.
- Displays the predicted class with **matplotlib**.

## Technologies Used

- **Python**
- **TensorFlow/Keras**
- **OpenCV**
- **NumPy**
- **Matplotlib**
- **Google Colab** (for training and testing)

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- TensorFlow
- OpenCV
- NumPy
- Matplotlib

### Training the Model

To train the model on the CIFAR-10 dataset, run:

```bash
python train.py
