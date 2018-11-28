import tensorflow as tf
import keras.backend.tensorflow_backend as K
import numpy as np
import argparse
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.datasets import mnist
from keras.utils import np_utils
from mnist_add_rbf import add_rbf_layer
from keras.models import model_from_json
from keras.models import load_model
from rbflayer import RBFLayer

from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import SPSA
from cleverhans.dataset import MNIST
from cleverhans.loss import CrossEntropy
from cleverhans.train import train
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import cnn_model
from customWrapper import KerasModelWrapper
from cleverhans.utils_tf import model_eval
from cleverhans.evaluation import batch_eval
from cleverhans.attacks_tf import fgsm


def eval(sess, model_name, X_train, Y_train, X_test, Y_test, cnn=False, rbf=False):
    """ Load model saved in model_name.json and model_name_weights.h5 and 
    evaluate its accuracy on legitimate test samples and adversarial samples.
    Use cnn=True if the model is CNN based.
    """

    # load saved model
    print("Load model ... ")
    '''
    json = open('models/{}.json'.format(model_name), 'r')
    model = json.read()
    json.close()
    loaded_model = model_from_json(model)
    loaded_model.load_weights("models/{}_weights.h5".format(model_name))
    '''
    if rbf:
        loaded_model = load_model("rbfmodels/{}.h5".format(model_name), custom_objects={'RBFLayer': RBFLayer})
    else:
        loaded_model = load_model("models/{}.h5".format(model_name))

    # Set placeholders
    if cnn:
        x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    else:
        x = tf.placeholder(tf.float32, shape=(None, 784))

    y = tf.placeholder(tf.float32, shape=(None, 10))

    predictions = loaded_model(x)

    accuracy = model_eval(sess, x, y, predictions, X_test, Y_test, args={ "batch_size" : 128 })
    print('Test accuracy on legitimate test examples: ' + str(accuracy))

    # Craft adversarial examples using Fast Gradient Sign Method (FGSM)
    # Using functions from /cleverhans/attacks_tf.py
    # Will be deprecated next year
    # adv_x = fgsm(x, predictions, eps=0.3)
    # X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test], batch_size=128)

    # Using functions from /cleverhans/attacks.py (as specified by creators)

    wrap = KerasModelWrapper(loaded_model)
    spsa = SPSA(wrap, sess=sess)

    images = 100

    correctImages = 0
    adv_pred = np.zeros((images,10))

    for i in range(images):
        tensorpls = X_test[i].reshape(1,784)
        tensorpls2 = Y_test[i].reshape(1,10)

        x_in = tf.convert_to_tensor(tensorpls, tf.float32)
        y_in = tf.convert_to_tensor(tensorpls2, tf.float32)
        
        adv_x = spsa.generate(x_in, y_in, eps = 0.3, nb_iter = 100, clip_min = 0, clip_max = 1, early_stop_loss_threshold=-1., spsa_samples=32, spsa_iters=1)
        adv_x = tf.stop_gradient(adv_x)

        test2 = adv_x.eval(session = sess)
        test3 = test2.reshape(28,28)
        plt.imshow(test3)
        plt.colorbar()
        plt.show()
        
        print (type(test2))
        print (test2.shape)
        
        preds_adv = loaded_model(adv_x)

        test = preds_adv.eval(session = sess)

        for j in range(10):
            adv_pred[i][j] = test[0][j]

        if np.argmax(adv_pred[i]) == np.argmax(Y_test[i]):
            correctImages = correctImages + 1

        accuracy = correctImages / (i + 1)
        print('Test accuracy (' + str(i + 1) + '): ' + str(accuracy))
            
    # Evaluate the accuracy of the MNIST model on adversarial examples
    #accuracy = model_eval(sess, x, y, preds_adv, X_test, Y_test, args={ "batch_size" : 128 })

    accuracy = correctImages / images
    print('Test accuracy on adversarial test examples: ' + str(accuracy))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', metavar='model_name', type=str,
                        help='model saved in model_name.json and model_name_weights.h5')
    parser.add_argument('--cnn', action='store_true', help='cnn type network (2d input)')
    parser.add_argument('--rbf', action='store_true', help='true if model contains rbf layer')
    
    args = parser.parse_args()
    model_name= args.model_name
    cnn = args.cnn
    rbf = args.rbf

    # Setup a TF session
    if not hasattr(K, "tf"):
        raise RuntimeError("This python file requires keras to be configured"
                       " to use the TensorFlow backend.")

    sess = tf.Session()
    K.set_session(sess)
    
    # Get MNIST test data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print("Loaded MNIST test data.")

    if cnn:
        X_train = X_train.reshape(60000, 28, 28, 1)
        X_test = X_test.reshape(10000, 28, 28, 1)
    else:
        X_train = X_train.reshape(60000, 784)
        X_test = X_test.reshape(10000, 784)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    eval(sess, model_name, X_train, Y_train, X_test, Y_test, cnn, rbf)
