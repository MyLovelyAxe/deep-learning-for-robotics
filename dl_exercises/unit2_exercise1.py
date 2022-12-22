import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

data = pd.read_csv(
    '~/catkin_ws/src/dlrepo/velocity_data.csv', sep=',', header=0)

t = data['field.header.stamp'].astype(float)
t = t / data['field.header.stamp'].iloc[0]

X = data['field.pose.pose.position.x'].astype(float)


def NN(t, X):

    model = Sequential()

    #######################################   HIDDEN LAYER   #################################################
    # a dense layer (fuly connected layer) with 3 neurons
    # as hidden layer
    model.add(Dense(units=3, input_shape=(1,), use_bias=True))
    #######################################   OUTPUT LAYER   ##################################################
    # a dense layer (fully connected layer) with 1 neuron
    # as output layer
    model.add(Dense(1))
    ##########################################################################################################

    # In the shell, view a summary of the network you've built:
    model.summary()
    # Define an optimizer, a loss function, and the metric you want to minimize.
    model.compile('rmsprop', loss='mse', metrics=['mse'])
    # Define the training phase, specifying how many iterations or epochs you want it to run. Also, call the function to start training:
    h = model.fit(t, X, epochs=500, verbose=True)

    return model


model = NN(t, X)

# Build a prediction set to store the output that the model predicts. Store it in a DataFrame:
t_pred = np.arange(t.iloc[-1], t.iloc[-1]+1.0, 0.1)
X_pred = model.predict(t_pred)
pred_data = pd.DataFrame()
pred_data['t'] = t_pred
pred_data['X'] = X_pred

# Plot the original data:
original_data = pd.DataFrame()
original_data['t'] = t
original_data['X'] = X
ax = original_data.plot(x='t', y='X', style='o')

# Plot the predicted data:
pred_data.plot(ax=ax, x='t', y='X', style='rx')
plt.show()
