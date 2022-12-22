from utilities import load_dataframe
from utilities import plot_points_classes
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import tensorflow as tf
import numpy as np
import pandas as pd

dataset = load_dataframe(
    '/home/user/catkin_ws/src/dlrepo/dl_exercises/position_data.csv')
input_coordinates = dataset[:, :-1]
label = dataset[:, -1]

X, Y = input_coordinates, label


def NN(X, Y):
    model = Sequential()
    # hidden layer
    model.add(Dense(units=2, input_shape=(2,), use_bias=True))
    # output layer
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(optimizer='sgd', loss='binary_crossentropy',
                  metrics=['accuracy'])
    h = model.fit(X, Y, epochs=50, verbose=False)

    plt.plot(h.history['accuracy'])
    plt.ylabel('Accuracy evolution')
    plt.xlabel('Epochs')
    plt.show()
    return model


def predict(model, x):

    # output y_pred: (3,1) with 3 probabilities
    y_prob = model.predict(x)
    print("y_prob: ")
    print(y_prob, y_prob.shape)
    # make the probabilities as 0 when it is less than 0.5
    # make the probabilities as 1 when it is more than 0.5
    # y_pred contains labels
    y_pred = np.where(y_prob < 0.5, np.zeros_like(
        y_prob), np.ones_like(y_prob)).reshape(len(y_prob))
    print("y_pred: ")
    print(y_pred, y_pred.shape)
    # plot the points
    print(x)
    plot_points_classes(x, y_pred)


model = NN(X, Y)

# input x: (3,2) 3 points with 2 dimension
x = np.array([[7, 6], [4, 7], [4, 2]])
predict(model, x)
