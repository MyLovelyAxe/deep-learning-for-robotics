import sys
sys.path.append('~/catkin_ws/src/dlrepo/utilities')
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from utilities import load_dataframe

dataset = load_dataframe('/home/user/catkin_ws/src/dlrepo/dl_exercises/position_data.csv')
input_coordinates = dataset[:,:-1]
label = dataset[:,-1]

X,Y = input_coordinates,label

def NN(X,Y):
    model = Sequential()
    model.add(Dense(units=2, input_shape=(2,),use_bias=True))
    model.add(Dense(1,activation='sigmoid'))
    model.summary()
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    h = model.fit(X,Y, epochs=50,verbose=False)
    
    plt.plot(h.history['accuracy'])
    plt.ylabel('Accuracy evolution')
    plt.xlabel('Epochs')
    plt.show()
    return model

model = NN(X,Y)