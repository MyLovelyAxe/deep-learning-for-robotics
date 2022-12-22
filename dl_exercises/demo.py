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
    model.add(Dense(units=1, input_shape=(1,), use_bias=True))
    #######################################   OUTPUT LAYER   ##################################################
    model.add(Dense(1))
    ##########################################################################################################

    model.summary()
    model.compile('rmsprop', loss='mse', metrics=['mse'])
    h = model.fit(t, X, epochs=500, verbose=True)
    return model


model = NN(t, X)

t_pred = np.arange(t.iloc[-1], t.iloc[-1]+1.0, 0.1)
X_pred = model.predict(t_pred)
pred_data = pd.DataFrame()
pred_data['t'] = t_pred
pred_data['X'] = X_pred

original_data = pd.DataFrame()
original_data['t'] = t
original_data['X'] = X

ax = original_data.plot(x='t', y='X', style='o')
pred_data.plot(ax=ax, x='t', y='X', style='rx')
plt.show()
