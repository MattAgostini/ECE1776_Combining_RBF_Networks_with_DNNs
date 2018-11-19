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

from cleverhans.attacks import FastGradientMethod, SaliencyMapMethod
from cleverhans.attacks import DeepFool, ProjectedGradientDescent
from cleverhans.dataset import MNIST
from cleverhans.loss import CrossEntropy
from cleverhans.train import train
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import cnn_model
from customWrapper import KerasModelWrapper
from cleverhans.utils_tf import model_eval
from cleverhans.evaluation import batch_eval
from cleverhans.attacks_tf import fgsm


def eval(sess, model_name, X_train, Y_train, X_test, Y_test, cnn=False, rbf=False, fgsm=False, jsma=False, df=False, bim=False):
    """ Load model saved in model_name.json and model_name_weights.h5 and 
    evaluate its accuracy on legitimate test samples and adversarial samples.
    Use cnn=True if the model is CNN based.
    """

    # open text file and output accuracy results to it
    text_file = open("cifar_results.txt", "w")

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
        text_file.write('Evaluating on rbfmodels/{}.h5\n\n'.format(model_name))
    else:
        loaded_model = load_model("models/{}.h5".format(model_name))
        text_file.write('Evaluating on models/{}.h5\n\n'.format(model_name))

    # Set placeholders
    if cnn:
        x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    else:
        x = tf.placeholder(tf.float32, shape=(None, 3072))

    y = tf.placeholder(tf.float32, shape=(None, 10))

    predictions = loaded_model(x)

    accuracy = model_eval(sess, x, y, predictions, X_test, Y_test, args={ "batch_size" : 128 })
    text_file.write('Test accuracy on legitimate test examples: {0}\n'.format(str(accuracy)))
    #print('Test accuracy on legitimate test examples: ' + str(accuracy))

    # Craft adversarial examples depending on the input parameters
    wrap = KerasModelWrapper(loaded_model)
    
    # FGSM
    if fgsm:
        fgsm = FastGradientMethod(wrap, sess=sess)
        fgsm_params = {'eps': 0.3}
        adv_x = fgsm.generate(x, **fgsm_params)
        adv_x = tf.stop_gradient(adv_x)
        preds_adv = loaded_model(adv_x)

        # Evaluate the accuracy of the CIFAR-10 model on adversarial examples
        accuracy = model_eval(sess, x, y, preds_adv, X_test, Y_test, args={ "batch_size" : 128})
        text_file.write('Test accuracy on fgsm adversarial test examples: {0}\n'.format(str(accuracy)))
        #print('Test accuracy on fgsm adversarial test examples: ' + str(accuracy))

    # JSMA
    if jsma:
        jsma = SaliencyMapMethod(wrap, sess=sess)
        jsma_params = {'theta': 2., 'gamma': 0.145,
                   'clip_min': 0., 'clip_max': 1.,
                   'y_target': None}
        adv_x = jsma.generate(x, **jsma_params)
        adv_x = tf.stop_gradient(adv_x)
        preds_adv = loaded_model(adv_x)

        # Evaluate the accuracy of the CIFAR-10 model on adversarial examples
        accuracy = model_eval(sess, x, y, preds_adv, X_test, Y_test, args={ "batch_size" : 128})
        text_file.write('Test accuracy on jsma adversarial test examples: {0}\n'.format(str(accuracy)))
        #print('Test accuracy on jsma adversarial test examples: ' + str(accuracy))

    # DeepFool
    if df:
        df = DeepFool(wrap, sess=sess)
        df_params = {'nb_candidate': 10,
                 'max_iter': 50}
        adv_x = df.generate(x, **df_params)
        adv_x = tf.stop_gradient(adv_x)
        preds_adv = loaded_model(adv_x)

        # Evaluate the accuracy of the CIFAR-10 model on adversarial examples
        accuracy = model_eval(sess, x, y, preds_adv, X_test, Y_test, args={ "batch_size" : 128})
        text_file.write('Test accuracy on df adversarial test examples: {0}\n'.format(str(accuracy)))
        #print('Test accuracy on df adversarial test examples: ' + str(accuracy))

    # Basic Iterative Method
    # Commented out as it is hanging on batch #0 at the moment
    '''
    if bim:
        bim = ProjectedGradientDescent(wrap, sess=sess)
        bim_params = {'eps': 0.3}
        adv_x = bim.generate(x, **bim_params)
        adv_x = tf.stop_gradient(adv_x)
        preds_adv = loaded_model(adv_x)

        # Evaluate the accuracy of the CIFAR-10 model on adversarial examples
        accuracy = model_eval(sess, x, y, preds_adv, X_test, Y_test, args={ "batch_size" : 128})
        text_file.write('Test accuracy on bim adversarial test examples: {0}\n'.format(str(accuracy)))
        #print('Test accuracy on bim adversarial test examples: ' + str(accuracy))
    '''
    print('Accuracy results outputted to cifar10_results.txt')
    text_file.close()

    # Close TF session
    sess.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', metavar='model_name', type=str,
                        help='model saved in model_name.json and model_name_weights.h5')
    parser.add_argument('--cnn', action='store_true', help='cnn type network (2d input)')
    parser.add_argument('--rbf', action='store_true', help='true if model contains rbf layer')
    parser.add_argument('--fgsm', action='store_true', help='run fgsm adversarial attack and get accuracy results')
    parser.add_argument('--jsma', action='store_true', help='run jsma adversarial attack and get accuracy results')
    parser.add_argument('--df', action='store_true', help='run deepfool adversarial attack and get accuracy results')
    parser.add_argument('--bim', action='store_true', help='run basic iterative method adversarial attack and get accuracy results')
    
    args = parser.parse_args()
    model_name= args.model_name
    cnn = args.cnn
    rbf = args.rbf
    fgsm = args.fgsm
    jsma = args.jsma
    df = args.df
    bim = args.bim

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

    eval(sess, model_name, X_train, Y_train, X_test, Y_test, cnn, rbf, fgsm, jsma, df, bim)
