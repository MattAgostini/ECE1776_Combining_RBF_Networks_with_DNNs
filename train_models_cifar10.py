import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.datasets import mnist, cifar10
from keras.utils import np_utils
from keras.optimizers import SGD
from keras import regularizers
from keras.applications.vgg19 import VGG19

num_classes = 10

def mlp(model_name, x_train, y_train, x_test, y_test):
    mlp = Sequential()
    mlp.add(Dense(512, input_dim=3072, activation='relu'))
    mlp.add(Dense(512, activation='relu'))
    mlp.add(Dense(10, activation='softmax'))

    mlp.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    mlp.fit(x_train, y_train,
              epochs=100,
              batch_size=128,
              validation_data=(x_test, y_test))
    #score = mlp.evaluate(x_test, y_test, batch_size=128)
    mlp.save("models/{}_cifar10.h5".format(model_name))

def cnn(model_name, x_train, y_train, x_test, y_test):
    cnn = Sequential()

    cnn.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                          input_shape=(32,32,3)))

    cnn.add(Conv2D(64, (3, 3), activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Dropout(0.25))
    cnn.add(Flatten())
    cnn.add(Dense(128, activation='relu'))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(num_classes, activation='softmax'))

    cnn.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    cnn.fit(x_train, y_train,
              epochs=100,
              batch_size=128,
              validation_data=(x_test, y_test),
              verbose=0)
    score = cnn.evaluate(x_test, y_test, batch_size=128)
    print("Test accuracy: ", score)
    cnn.save("models/{}_cifar10.h5".format(model_name))

def vgg19(model_name, x_train, y_train, x_test, y_test):
    model = VGG19(weights=None, input_shape=(32,32,3), classes=10)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=200,
              batch_size=128)
    score = model.evaluate(x_test, y_test, batch_size=128)
    model.save("models/{}.h5".format(model_name))

def vgg16(model_name, x_train, y_train, x_test, y_test):
    model = VGG16(weights=None, input_shape=(32,32,3), classes=10)

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print("Loaded CIFAR-10 test data.")

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0

    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)
    
    #mlp('mlp_test', X_train.reshape(50000, 3072), Y_train, X_test.reshape(10000, 3072), Y_test)
    cnn('cnn_test', X_train.reshape(50000, 32, 32, 3), Y_train, X_test.reshape(10000, 32, 32, 3), Y_test)

    #vgg16('vgg16_cifar10', X_train.reshape(50000, 32, 32, 3), Y_train, X_test.reshape(10000, 32, 32, 3), Y_test)
    #vgg19('vgg19_cifar10', X_train.reshape(50000, 32, 32, 3), Y_train, X_test.reshape(10000, 32, 32, 3), Y_test)
    
