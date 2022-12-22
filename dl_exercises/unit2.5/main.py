#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from planar_utils import load_planar_dataset, plot_dataset, plot_decision_boundary
from helpers import sigmoid, sigmoid_backward, relu, relu_backward
import utilities

#######################
###### load data ######
#######################


def load_data():

    # X: (2, 400) <class 'numpy.ndarray'>
    # Y: (1, 400) <class 'numpy.ndarray'>
    X, Y = load_planar_dataset()
    print("input data X: ")
    print(X.shape, type(X), "\n")
    print("label Y: ")
    print(Y.shape, type(Y), "\n")

    # plot_dataset(X, Y)

    # np.where(Y==0):
    # return: tuple: (value_matrix, index_matrix)
    X_label_0 = X[:, np.where(Y == 0)[1]]
    X_label_1 = X[:, np.where(Y == 1)[1]]

    # show data
    # plt.figure(figsize=(12, 8))
    # plt.scatter(X_label_0[0], X_label_0[1], color='red', label='label is 0')
    # plt.scatter(X_label_1[0], X_label_1[1], color='blue', label='label is 1')
    # plt.xlabel("x-coordinate")
    # plt.ylabel("y-coordinate")
    # plt.title("show dataset with position coordinates and labels")
    # plt.legend()
    # plt.show()

    return X, Y

###############################################
###### Define a Neural Network Structure ######
###############################################


def get_parameters(X, Y):

    # The number of examples in the dataset
    m = X.shape[1]
    # The number of coordinates in the input set, which will be the number of neurons in the input layer
    # for one single sample
    n_coords = X.shape[0]
    # The number of outputs in the train set, which will be the number of neurons in the output layer
    # for one single sample
    n_outputs = Y.shape[0]

    print("There are {} examples in the dataset".format(m), "\n")
    print("There are {} coordinates in the input set".format(n_coords), "\n")
    print("There are {} outputs in the train set".format(n_outputs), "\n")

    return m, n_coords, n_outputs

################################################
###### Initialize Parameters of the Model ######
################################################


def init_parameters(layer_sizes):

    parameters = {}
    scale = 0.001

    start = 1
    prev_h = layer_sizes[0]

    for idx, h in enumerate(layer_sizes[1:]):

        now_h = h
        number = prev_h * now_h
        end = start + number

        parameters['w'+str(idx+1)] = np.linspace(start,
                                                 end-1, number).reshape(prev_h, now_h) * scale
        parameters['b'+str(idx+1)] = np.zeros((now_h))

        start = end
        prev_h = now_h

        print("weights_{}: ".format(idx+1))
        print(parameters['w'+str(idx+1)],
              parameters['w'+str(idx+1)].shape, "\n")
        print("bias_{}: ".format(idx+1))
        print(parameters['b'+str(idx+1)],
              parameters['b'+str(idx+1)].shape, "\n")

    return parameters

#################################
###### Forward Propagation ######
#################################


def forward_activation(A_prev, W, b, acti_func):

    Z = np.dot(A_prev, W) + b

    if acti_func == "sigmoid":
        A = sigmoid(Z)
    elif acti_func == "relu":
        A = relu(Z)

    return (A, W, b, Z)


def forward_propagation(parameters, X):

    weights = []
    bias = []

    for idx, para in enumerate(parameters):

        if (idx+2) % 2 == 0:
            weights.append(parameters[para])
        else:
            bias.append(parameters[para])

    print("weights: ")
    print(weights, "\n")
    print("bias: ")
    print(bias, "\n")

    # to store tuples (A_prev, w, b, Z) of each layer
    caches = []
    A_prev = X.T

    for idx, (w, b) in enumerate(zip(weights, bias)):

        if idx != (len(weights)-1):
            (A, w, b, Z) = forward_activation(A_prev, w, b, "relu")
            A_prev = A
            caches.append((A, w, b, Z))
            print("A{}: ".format(idx+1))
            print(A.shape, "\n")
        else:
            break

    (A_label, w, b, Z) = forward_activation(A_prev, w, b, "sigmoid")
    caches.append((A_label, w, b, Z))
    A_label = np.squeeze(A_label)
    print("A_label: ")
    print(A_label.shape, "\n")

    return A_label, caches

############################
###### Calculate Loss ######
############################


def calculate_cost(A_label, Y, m):

    loss = np.sum(- np.multiply(Y, np.log(A_label)) -
                  np.multiply((1 - Y), np.log(1 - A_label))) / m
    print("loss of this epoch: ", loss)

    return loss

#############################################################
###### Implement Backward Propagation to Get Gradients ######
#############################################################


def backward_activation_output(Y, caches, m):

    # A_label: (400,)
    # w2: (3,1)
    # b2: (1,)
    # Z2: (400,)
    A_label, w2, b2, Z2 = caches[-1]
    # A_prev: (400,3)
    # w1: (2,3)
    # b1: (3,)
    # Z1: (400,3)
    A_prev, w1, b1, Z1 = caches[-2]
    # dA_out: (400,)
    dA_out = np.sum(- (Y / A_label) + (1-Y) / (1 - A_label)) / m
    # dz_out: (400,)
    dz_out = sigmoid_backward(dA_out, Z2)
    # dw_out = dz_out * A_prev = (400,) * (400,3) = (3,)
    dw_out = np.dot(A_prev.T, dz_out.reshape(len(A_prev), 1))
    print("dw_out shape: ")
    print(dw_out.shape, "\n")


def backward_propagation():
    pass


def main():

    # load data
    X, Y = load_data()
    # get parameters for constructure of NN
    m, n_coords, n_outputs = get_parameters(X, Y)
    # a list storing the number of neurons for each layer: input layer, hidden layer, output layer
    # which can be customized
    layer_sizes = [n_coords, 3, n_outputs]
    # get weights matrix and bias matrix
    parameters = init_parameters(layer_sizes)
    # forward propagation
    A_label, caches = forward_propagation(parameters, X)
    # calculate cross-entropy loss
    loss = calculate_cost(A_label, Y, m)

    backward_activation_output(Y, caches, m)


if __name__ == "__main__":

    main()
