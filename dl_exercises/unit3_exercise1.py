import sys
sys.path.append('~/catkin_ws/src/dlrepo/utilities')
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from utilities import load_dataframe

dataset = load_dataframe('/home/user/ai_ws/position_data.csv')
input_coordinates = dataset[:,:-1]
label = dataset[:,-1]

#X,Y = input_coordinates,label

from sklearn.model_selection import train_test_split


X_train,X_val,Y_train,Y_val = train_test_split(input_coordinates,label,test_size=0.20,random_state=45)

len(X_train), len(X_val)


def NN(X_train,X_val,Y_train,Y_val):
    model = Sequential()
    model.add(Dense(units=2, input_shape=(2,),use_bias=True))
    model.add(Dense(1,activation='sigmoid'))
    model.summary()
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    h = model.fit(X_train,Y_train, validation_data=(X_val,Y_val), epochs=50,verbose=True)
    
    plt.plot(h.history['accuracy'])
    plt.ylabel('Accuracy evolution')
    plt.xlabel('Epochs')
    plt.show()
    return model

model = NN(X_train,X_val,Y_train,Y_val)