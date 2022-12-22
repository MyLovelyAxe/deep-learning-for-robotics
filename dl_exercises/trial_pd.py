import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

data = pd.read_csv(
    '~/catkin_ws/src/dlrepo/velocity_data.csv', sep=',', header=0)

t = data['field.header.stamp'].astype(float)
print(t, t.shape, type(t))

t_iloc_0 = data['field.header.stamp'].iloc[0]
print(t_iloc_0, t_iloc_0.shape, type(t_iloc_0))

# t = t / data['field.header.stamp'].iloc[0]

# X = data['field.pose.pose.position.x'].astype(float)
