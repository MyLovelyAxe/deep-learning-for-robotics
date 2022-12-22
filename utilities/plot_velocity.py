import pandas as pd
from utilities import plot_points

df = pd.read_csv(
    '/home/user/catkin_ws/src/dlrepo/utilities/velocity_data_2.csv', sep=',')
t = df['field.header.stamp'].astype(float)
t = t / df['field.header.stamp'].iloc[0]
X = df['field.pose.pose.position.x'].astype(float)

# plot the points
plot_points(t, X, 'X(t)')
