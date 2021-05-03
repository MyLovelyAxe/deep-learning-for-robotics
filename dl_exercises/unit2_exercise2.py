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

# do not run
from utilities import plot_points_classes_predicted

#PREDICT
X_pred = np.array([[7,6],[4,7],[4,2]])
output_pred = model.predict(X_pred)
Y_pred=[]
for labels in output_pred:
    if labels<=0.5:
        Y_pred.append(0.0)
    else: Y_pred.append(1.0)
Y_pred = np.array(Y_pred)

#PLOT
plot_points_classes_predicted(X,Y,X_pred,Y_pred)