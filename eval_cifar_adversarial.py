import tensorflow as tf
import keras.backend.tensorflow_backend as K
import numpy as np
import argparse
from keras.models import model_from_json
from keras.datasets import mnist, cifar10
from keras.utils import np_utils
from mnist_add_rbf import add_rbf_layer
from keras.models import model_from_json
from keras.models import load_model
from rbflayer import RBFLayer

from cleverhans.attacks import FastGradientMethod
from cleverhans.dataset import MNIST
from cleverhans.loss import CrossEntropy
from cleverhans.train import train
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import cnn_model
from cleverhans.utils_keras import KerasModelWrapper
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
        x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    else:
        x = tf.placeholder(tf.float32, shape=(None, 3072))

    y = tf.placeholder(tf.float32, shape=(None, 10))

    predictions = loaded_model(x)

    accuracy = model_eval(sess, x, y, predictions, X_test, Y_test, args={ "batch_size" : 128 })
    print('Test accuracy on legitimate test examples: ' + str(accuracy))

    # Craft adversarial examples using Fast Gradient Sign Method (FGSM)
    # Using functions from /cleverhans/attacks_tf.py
    # Will be deprecated next year
    adv_x = fgsm(x, predictions, eps=0.3)
    X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test], batch_size=128)

    # Using functions from /cleverhans/attacks.py (as specified by creators)
    # Does not work at the moment
    '''
    wrap = KerasModelWrapper(loaded_model)
    fgsm = FastGradientMethod(wrap, sess=sess)
    fgsm_params = {'eps': 0.3}
                   #'y': y}
    adv_x = fgsm.generate(x, **fgsm_params)
    adv_x = tf.stop_gradient(adv_x)
    X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test], batch_size=128)
    predictions_adv = loaded_model(adv_x)
    '''
    
    # Evaluate the accuracy of the MNIST model on adversarial examples
    accuracy = model_eval(sess, x, y, predictions, X_test_adv, Y_test, args={ "batch_size" : 128 })
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
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print("Loaded CIFAR-10 test data.")

    if cnn:
        X_train = X_train.reshape(50000, 32, 32, 3)
        X_test = X_test.reshape(10000, 32, 32, 3)
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
