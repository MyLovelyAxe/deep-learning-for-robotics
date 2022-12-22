from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2


def build_model():

    # initialize model
    model = Sequential()

    # feature detector
    model.add(Conv2D(32, (3, 3), activation='relu',
              padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    # classifier
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # compiler
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def train_model(model, X_train, y_train, X_val, y_val, n_epochs):

    # h: store the history progress of your training process
    h = model.fit(X_train, y_train, epochs=n_epochs, batch_size=64,
                  validation_data=(X_val, y_val))  # , verbose=0)

    return h


def save_model(model, path):
    model.save(path+'cnn_model.h5')
