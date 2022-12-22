from sklearn.model_selection import train_test_split
from utilities import load_dataframe
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow.keras as keras
# Improvement1: Regularization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
# Improvement2: Weight Initialization
from tensorflow.keras import initializers
# Improvement4: Momentum
from tensorflow.keras import optimizers

print(tf.__version__)
print(keras.__version__)

dataset = load_dataframe(
    '/home/user/catkin_ws/src/dlrepo/dl_exercises/position_data.csv')
input_coordinates = dataset[:, :-1]
label = dataset[:, -1]

#X,Y = input_coordinates,label


X_train, X_val, Y_train, Y_val = train_test_split(
    input_coordinates, label, test_size=0.3, random_state=45)

print("the shape of X_train and X_val: ")
print(X_train.shape, X_val.shape)


def NN(X_train, X_val, Y_train, Y_val):
    model = Sequential()
    model.add(Dense(units=2, input_shape=(2,), use_bias=True))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(optimizer='sgd', loss='binary_crossentropy',
                  metrics=['accuracy'])
    # verbose = True to show trainning information of each epoch
    h = model.fit(X_train, Y_train, validation_data=(
        X_val, Y_val), epochs=50, verbose=True)

    plt.plot(h.history['accuracy'])
    plt.ylabel('Accuracy evolution')
    plt.xlabel('Epochs')
    plt.show()
    return model


##########################################
###### Improvement1: Regularization ######
##########################################


def regularized_NN(X_train, X_val, Y_train, Y_val):

    lambda_value = 0.005
    dropout_prob = 0.1

    model = Sequential()
    # set l2-regularizer with lambda
    model.add(Dense(units=2, input_shape=(2,), use_bias=True,
              kernel_regularizer=l2(lambda_value)))
    model.add(Dense(1, activation='sigmoid'))
    # set dropout for layers
    # In the Keras implementation, the dropout will be added after the hidden layer declaration:
    model.add(Dropout(dropout_prob))
    model.summary()
    model.compile(optimizer='sgd', loss='binary_crossentropy',
                  metrics=['accuracy'])
    # verbose = True to show trainning information of each epoch
    h = model.fit(X_train, Y_train, validation_data=(
        X_val, Y_val), epochs=150, verbose=True)

    plt.plot(h.history['accuracy'])
    plt.ylabel('Accuracy evolution')
    plt.xlabel('Epochs')
    plt.show()
    return model

#################################################
###### Improvement2: Weight Initialization ######
#################################################


def weight_initialized_NN(X_train, X_val, Y_train, Y_val):

    initializer_1 = initializers.RandomNormal(mean=0., stddev=1.)
    initializer_2 = initializers.HeNormal()
    initializer_3 = initializers.GlorotUniform()

    model = Sequential()
    model.add(Dense(2, input_shape=(2,), activation='sigmoid',
              kernel_initializer=initializer_1))
    model.add(Dense(3, activation='tanh', kernel_initializer=initializer_2))
    model.add(Dense(1, activation='relu', kernel_initializer=initializer_3))

    model.summary()
    model.compile(optimizer='sgd', loss='binary_crossentropy',
                  metrics=['accuracy'])
    # verbose = True to show trainning information of each epoch
    h = model.fit(X_train, Y_train, validation_data=(
        X_val, Y_val), epochs=50, verbose=True)

    plt.plot(h.history['accuracy'])
    plt.ylabel('Accuracy evolution')
    plt.xlabel('Epochs')
    plt.show()
    return model

######################################
###### Improvement3: Mini-Batch ######
######################################


def Mini_Batch_NN(X_train, X_val, Y_train, Y_val):
    model = Sequential()
    model.add(Dense(units=2, input_shape=(2,), use_bias=True))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(optimizer='sgd', loss='binary_crossentropy',
                  metrics=['accuracy'])
    # batch gradient descent: the whole dataset at once as usual
    #    batch_size = m(whole number of samples)
    # stochastic gradient descent(sgd): one sample by one sample
    #    batch_size = 1
    # mini-batch: define how many samples constitute one mini-batch
    #    batch_size = n (1<n<m)
    # verbose = True to show trainning information of each epoch
    h = model.fit(X_train, Y_train, batch_size=32, validation_data=(
        X_val, Y_val), epochs=50, verbose=True)

    plt.plot(h.history['accuracy'])
    plt.ylabel('Accuracy evolution')
    plt.xlabel('Epochs')
    plt.show()
    return model

####################################
###### Improvement4: Momentum ######
####################################


def momentum_NN(X_train, X_val, Y_train, Y_val):

    model = Sequential()
    model.add(Dense(units=2, input_shape=(2,), use_bias=True))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    # parameter of 'momentum' in 'optimizers.SGD()' denotes the importance of previous values
    sgd_optimizer = optimizers.SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=sgd_optimizer, loss='binary_crossentropy',
                  metrics=['accuracy'])
    # verbose = True to show trainning information of each epoch
    h = model.fit(X_train, Y_train, validation_data=(
        X_val, Y_val), epochs=50, verbose=True)

    plt.plot(h.history['accuracy'])
    plt.ylabel('Accuracy evolution')
    plt.xlabel('Epochs')
    plt.show()
    return model


# model = NN(X_train, X_val, Y_train, Y_val)
# regularized_NN = regularized_NN(X_train, X_val, Y_train, Y_val)
# w_init_model = weight_initialized_NN(X_train, X_val, Y_train, Y_val)
# mini_batch_model = Mini_Batch_NN(X_train, X_val, Y_train, Y_val)
momentum_model = momentum_NN(X_train, X_val, Y_train, Y_val)
