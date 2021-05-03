import matplotlib.pyplot as plt
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


def load_dataset():
	# load dataset
	(X_train, y_train), (X_test, y_test) = cifar10.load_data()
	X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=0.20,random_state=45)
	# print size
	print('Loaded dataset with size: train (X,y): ',(X_train.shape,y_train.shape),',validation (X,y): ',(X_val.shape,y_val.shape),' and test (X,y):',(X_test.shape,y_test.shape))

	return X_train,y_train,X_val,y_val,X_test,y_test

def plot_dataset(X_train,y_train):

	# plot example images
	idx = [15,2500,10000]
	fig,axes = plt.subplots(3)
	axes[0].imshow(X_train[idx[0]])
	axes[1].imshow(X_train[idx[1]])
	axes[2].imshow(X_train[idx[2]])
	plt.show()

	print('Labels: ',y_train[idx[0]],y_train[idx[1]],y_train[idx[2]])


def preprocess_dataset(X_train,y_train,X_val,y_val,X_test,y_test):
	# convert labels to categorical
	y_train = to_categorical(y_train)
	y_val = to_categorical(y_val)
	y_test = to_categorical(y_test)
	# convert pixel values to floats
	X_train = X_train.astype('float32')
	X_val = X_val.astype('float32')
	X_test = X_test.astype('float32')
	# normalize pixel values
	X_train = X_train / 255.0
	X_val = X_val / 255.0
	X_test = X_test / 255.0

	return X_train,y_train,X_val,y_val,X_test,y_test

X_train1,y_train1,X_val1,y_val1,X_test1,y_test1 = load_dataset()
X_train2,y_train2,X_val2,y_val2,X_test2,y_test2 = preprocess_dataset(X_train1,y_train1,X_val1,y_val1,X_test1,y_test1)
plot_dataset(X_train2,y_train2)