# Load the dataset (function load_dataset)
# Plot some examples (function plot_dataset)
# Preprocess the dataset to enter the NN (function preprocess_dataset)
# Build the NN model (function build_model)
# Train the NN model (function train_model)
# Obtain the final accuracy (function calc_accuracy) and print it
# Plot the training history (function plot_history)
# Save the model in a .h5 file (function save_model) inside the path: ~/catkin_ws/src/dlrepo/get_images/src/

import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
from dataset import load_dataset, plot_dataset, preprocess_dataset
from model import build_model, train_model, save_model
from evaluate import calc_accuracy, plot_history

#############################################
###### load data and show some samples ######
#############################################


def data_loading():

    X_train1, y_train1, X_val1, y_val1, X_test1, y_test1 = load_dataset()
    X_train2, y_train2, X_val2, y_val2, X_test2, y_test2 = preprocess_dataset(
        X_train1, y_train1, X_val1, y_val1, X_test1, y_test1)
    plot_dataset(X_train2, y_train2)

    return X_train2, y_train2, X_val2, y_val2, X_test2, y_test2

#############################
###### training models ######
#############################


def model_training(X_train, y_train, X_val, y_val, n_epochs):

    model = build_model()
    h = train_model(model, X_train, y_train, X_val, y_val, n_epochs)

    return model, h

#######################################
###### show accuracy and history ######
#######################################


def show_result(model, X, y, h):

    calc_accuracy(model, X, y)
    plot_history(h)


def main():

    # parameters
    save_path = '~/catkin_ws/src/dlrepo/get_images/src/'
    n_epochs = 50

    # load data
    X_train2, y_train2, X_val2, y_val2, X_test2, y_test2 = data_loading()
    # create models
    model, h = model_training(X_train2, y_train2, X_val2, y_val2, n_epochs)
    # test the accuracy and show history
    show_result(model, X_test2, y_test2, h)
    # save history
    save_model(model, save_path)


if __name__ == "__main__":

    main()
