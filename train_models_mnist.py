import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import mnist
from keras.utils import np_utils

def mlp(model_name, x_train, y_train, x_test, y_test):
    mlp = Sequential()
    mlp.add(Dense(512, input_dim=784, activation='relu'))
    mlp.add(Dense(521, activation='relu'))
    mlp.add(Dense(10, activation='softmax'))

    mlp.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    mlp.fit(x_train, y_train,
              epochs=20,
              batch_size=128)
    score = mlp.evaluate(x_test, y_test, batch_size=128)
    mlp.save("models/{}.h5".format(model_name))

def cnn(model_name, x_train, y_train, x_test, y_test):
    cnn = Sequential()
    cnn.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    cnn.add(Conv2D(32, (3, 3), activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Flatten())
    cnn.add(Dense(128, activation='relu'))
    cnn.add(Dense(10, activation='softmax'))

    cnn.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    cnn.fit(x_train, y_train,
              epochs=12,
              batch_size=128)
    score = cnn.evaluate(x_test, y_test, batch_size=128)
    cnn.save("models/{}.h5".format(model_name))

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print("Loaded MNIST test data.")

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    # mlp('mnist_mlp', X_train.reshape(60000, 784), Y_train, X_test.reshape(10000, 784), Y_test)
    cnn('mnist_cnn', X_train.reshape(60000, 28, 28, 1), Y_train, X_test.reshape(10000, 28, 28, 1), Y_test)

    
