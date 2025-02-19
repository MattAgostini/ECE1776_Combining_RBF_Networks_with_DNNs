import numpy as np
from keras.models import Sequential, load_model, clone_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop, SGD
from keras.datasets import mnist
from keras.utils import np_utils
from rbflayer import RBFLayer, InitCentersRandom
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf

def accuracy_score(y1, y2):
    assert y1.shape == y2.shape
        
    y1_argmax = np.argmax(y1, axis=1)
    y2_argmax = np.argmax(y2, axis=1)
    score = sum(y1_argmax == y2_argmax)
    return (float(score) / len(y1)) * 100.0
    

def add_rbf_layer(model, betas, X_train, Y_train, X_test, Y_test):
    """ Create a new model as a copy of model + RBF network. Train 
    it on [X_train, Y_train], reports its test accuracy and returns 
    the new model.
    """

    sess = tf.InteractiveSession()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
    newmodel = Sequential() 
    copymodel = Sequential()
    for i in range(len(model.layers)):
        newmodel.add(model.layers[i])
        copymodel.add(model.layers[i])
   
    
    #    for layer in newmodel.layers:
    #        layer.trainable = False


    rbflayer = RBFLayer(300, betas=betas)
    
    

    newmodel.add(rbflayer)
    newmodel.add(Dense(10, use_bias=False, name="dense_rbf"))
    newmodel.add(Activation('softmax', name="Activation_rbf"))


    newmodel.compile(loss='categorical_crossentropy',
                     optimizer=RMSprop(),
                     metrics=['acc'])

    newmodel.summary()
    rbf = newmodel.get_layer(index=-3)

    '''
    import pdb; pdb.set_trace()
    rbf = newmodel.get_layer(index=-3)
    print("Betas and centers before training:")
    old_betas = sess.run(rbf.betas)
    old_centers = sess.run(rbf.centers)
    print(sess.run(rbf.betas))
    print(sess.run(rbf.centers))
    '''
    #model.compile(loss='mean_squared_error',
    #              optimizer=SGD(lr=0.1, decay=1e-6))

    newmodel.fit(X_train, Y_train,
                 batch_size=128,
                 epochs=3,
                 verbose=1)

    print("Betas and centers after training:")
    new_betas = sess.run(rbf.betas)
    new_centers = sess.run(rbf.centers)

    print(sess.run(rbf.betas))
    print(sess.run(rbf.centers))

    drbf = newmodel.get_layer(name="dense_rbf")
    trained_weights = sess.run(drbf.weights)[0]
    trained_weights = trained_weights.T
    from collections import Counter
    important_weights = Counter()

    for i in range(10):
        important = np.argpartition(trained_weights[i], -50)[-50:]
        for j in important:
            important_weights[j] += 1

    import pdb; pdb.set_trace()
    top_30_units = important_weights.most_common(30)
    tb = np.array(new_betas[top_30_units[0][0]])
    tc = np.array([new_centers[top_30_units[0][0]]])
    for unit, count in top_30_units[1:]:
        tb = np.append(tb, new_betas[unit])
        tc = np.append(tc, [new_centers[unit]], axis=0)

    import pdb; pdb.set_trace()

    Y_pred = newmodel.predict(X_test)
    print("Test Accuracy: ", accuracy_score(Y_pred, Y_test)) 

    rbflayer = RBFLayer(30, betas=betas)
    copymodel.add(rbflayer)
    copymodel.add(Dense(10, use_bias=False, name="dense_rbf"))
    copymodel.add(Activation('softmax', name="Activation_rbf"))

    op1 = copymodel.layers[-3].betas.assign(tb)
    op2 = copymodel.layers[-3].centers.assign(tc)
    sess.run(op1)
    sess.run(op2)

    #newmodel.layers[-2].set_weights(t)
    copymodel.compile(loss='categorical_crossentropy',
                     optimizer=RMSprop(),
                     metrics=['acc'])

    copymodel.summary()
    copymodel.fit(X_train, Y_train,
                 batch_size=128,
                 epochs=3,
                 verbose=1)
    Y_pred = copymodel.predict(X_test)
    print(accuracy_score(Y_pred, Y_test))

    import pdb; pdb.set_trace()

    return newmodel 
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('input_model_name', metavar='input', type=str,
                        help='input model saved in input.json and input_weights.h5')
    parser.add_argument('output_model_name', metavar='output', type=str,
                        help='output model saved in output.json and output_weights.h5')
    parser.add_argument('--betas', type=float, help='initial value for betas')
    parser.add_argument('--cnn', action='store_true', help='cnn type network (2d input)')
    
    args = parser.parse_args()
    input_model_name= args.input_model_name
    output_model_name = args.output_model_name
    betas = args.betas if args.betas else 2.0
    cnn = args.cnn

    # load and transform mnist data 
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if cnn:
        X_train = X_train.reshape(60000, 28, 28, 1)
    else:     
        X_train = X_train.reshape(60000, 784)
    X_train = X_train.astype('float32')
    X_train /= 255

    if cnn:
        X_test = X_test.reshape(10000, 28, 28, 1)
    else:
        X_test = X_test.reshape(10000, 784)
    X_test = X_test.astype('float32')
    X_test /= 255
    
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    # load model from file
    model = load_model("models/{}.h5".format(input_model_name))

    # create and learn new model 
    newmodel = add_rbf_layer(model, betas, X_train, Y_train, X_test, Y_test)

    # save new model to file 
    newmodel.save("rbfmodels/{}.h5".format(output_model_name))

    print("---test----")
    m = load_model("rbfmodels/{}.h5".format(output_model_name), custom_objects={'RBFLayer': RBFLayer})

    Y_pred = m.predict(X_test)
    print("Test Accuracy: ", accuracy_score(Y_pred, Y_test)) 
