import matplotlib.pyplot as plt
import numpy as np
import csv


def load_dataframe(name_csv):
    with open(name_csv, 'r') as f:
        lines = csv.reader(f, delimiter=',')
        dataset = list(lines)
        for i in range(len(dataset)):
            if dataset[i][2] == 'True':
                dataset[i][2] = 0.0
            elif dataset[i][2] == 'False':
                dataset[i][2] = 1.0
            dataset[i] = [float(x) for x in dataset[i]]
    return np.array(dataset)

# dataset = load_dataframe('position_data.csv')
# input = dataset[:,:-1]
# label = dataset[:,-1]


def load_dataset(name_csv):
    with open(name_csv, 'r') as f:
        lines = csv.reader(f, delimiter=',')
        dataset = list(lines)
        Y = [0]*len(dataset)
        for i in range(len(dataset)):
            if dataset[i][2] == 'True':
                Y[i] = 1.0
            elif dataset[i][2] == 'False':
                Y[i] = 0.0
            dataset[i] = [float(x) for x in dataset[i][0:2]]
        X = np.array(dataset[:]).T
        Y = np.array(Y).T
        return X, Y


def plot_dataset(X, Y):
    plt.scatter(X[0, :], X[1, :], c=Y.reshape(X[0, :].shape[0],))
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def plot_points_classes(input, label):
    input_0 = input[np.where(label == 0.0)].reshape(-1, 2)
    input_1 = input[np.where(label == 1.0)].reshape(-1, 2)

    print("input_0: ")
    print(input_0, input_0.shape, type(input_0))

    print("input_1: ")
    print(input_1, input_1.shape, type(input_1))

    plt.scatter(input_0[:, 0], input_0[:, 1], c='b', label='label=False')
    plt.scatter(input_1[:, 0], input_1[:, 1], c='r', label='label=True')
    plt.xlabel('Coordinate X')
    plt.ylabel('Coordinate Y')
    plt.legend()
    plt.show()

# plot_points_classes(input,label)


def plot_points_classes_predicted(input, label, input_pred, label_pred):
    input_0 = input[np.where(label == 0.0)]
    input_1 = input[np.where(label == 1.0)]

    plt.scatter(input_0[:, 0], input_0[:, 1], c='b', label='label=False')
    plt.scatter(input_1[:, 0], input_1[:, 1], c='r', label='label=True')
    # print predictions
    for idx, predictions in enumerate(input_pred):
        if label_pred[idx] == 0.0:
            plt.scatter(predictions[0], predictions[1],
                        c='y', label='pred=False')
        elif label_pred[idx] == 1.0:
            plt.scatter(predictions[0], predictions[1],
                        c='g', label='pred=True')

    plt.xlabel('Coordinate X')
    plt.ylabel('Coordinate Y')
    plt.legend()
    plt.show()

# plot the points


def plot_points(x, f, name):
    fig, ax = plt.subplots()
    # ax.spines['left'].set_position('zero')
    # ax.spines['bottom'].set_position('zero')
    plt.xlabel('Values of t')
    plt.ylabel('Values of X')
    ax.plot(x, f, 'bo')
    ax.grid()
    plt.legend([name], loc='upper right')

    plt.show()
