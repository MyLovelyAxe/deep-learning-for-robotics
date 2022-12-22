from utilities import plot_points_classes, load_dataframe
import numpy as np

dataset = load_dataframe(
    '/home/user/catkin_ws/src/dlrepo/utilities/position_data_1.csv')
input_coordinates = dataset[:, :-1]
label = dataset[:, -1]

# plot the points
plot_points_classes(input_coordinates, label)
