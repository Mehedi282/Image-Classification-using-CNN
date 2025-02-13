# Image-Classification-using-CNN

CIFAR-10 Image Classification using CNN



Overview

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images from the CIFAR-10 dataset. The model is trained on CIFAR-10 and can predict the class of a given image uploaded by the user.

Features

Trains a CNN model on the CIFAR-10 dataset.

Saves and loads the trained model (cifar10_cnn_model.h5).

Allows users to upload an image for prediction.

Uses OpenCV for image preprocessing.

Displays the predicted class with matplotlib.

Technologies Used

Python

TensorFlow/Keras

OpenCV

NumPy

Matplotlib

Google Colab (for training and testing)

Installation

Prerequisites

Ensure you have the following installed:

Python 3.x

TensorFlow

OpenCV

NumPy

Matplotlib

Training the Model

To train the model on the CIFAR-10 dataset, run:

python train.py

This will train the model and save it as cifar10_cnn_model.h5.

Predicting an Image

Run the script and upload an image for prediction:

python predict.py

The script will:

Ask for an image upload.

Preprocess the image.

Predict the class.

Display the image with the predicted label.

Dataset

The model is trained on the CIFAR-10 dataset, which contains 60,000 images across 10 classes:

Airplane

Automobile

Bird

Cat

Deer

Dog

Frog

Horse

Ship

Truck

Contributing

Feel free to contribute by submitting issues or pull requests.

License

This project is licensed under the MIT License.
