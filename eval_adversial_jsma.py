import tensorflow as tf
import keras.backend.tensorflow_backend as K
import numpy as np
import argparse
from six.moves import xrange
from tensorflow.python.platform import flags
from keras.models import model_from_json
from keras.datasets import mnist
from keras.utils import np_utils
from mnist_add_rbf import add_rbf_layer
from keras.models import model_from_json
from keras.models import load_model
from rbflayer import RBFLayer

from cleverhans.utils import other_classes, set_log_level
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils_tf import model_argmax
from cleverhans.attacks import FastGradientMethod, SaliencyMapMethod
from cleverhans.dataset import MNIST
from cleverhans.loss import CrossEntropy
from cleverhans.train import train
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import cnn_model
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.utils_tf import model_eval
from cleverhans.evaluation import batch_eval
from cleverhans.attacks_tf import fgsm, jsma

FLAGS = flags.FLAGS

VIZ_ENABLED = True
NB_EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = .001
SOURCE_SAMPLES = 10


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

    # Craft adversarial examples using Jacobian-based Saliency Map Approach (JSMA)
    wrap = KerasModelWrapper(loaded_model)
    jsma = SaliencyMapMethod(wrap, sess=sess)
    jsma_params = {'theta': 1., 'gamma': 1,
                   'clip_min': 0., 'clip_max': 1.,
                   'y_target': None}
    adv_x = jsma.generate(x, **jsma_params)
    adv_x = tf.stop_gradient(adv_x)
    preds_adv = loaded_model(adv_x)

    accuracy = model_eval(sess, x, y, preds_adv, X_test, Y_test, args={ "batch_size" : 512 })
    print('Test accuracy on adversarial test examples: ' + str(accuracy))
    '''
    report = AccuracyReport()
    viz_enabled=VIZ_ENABLED
    source_samples=SOURCE_SAMPLES
    img_rows, img_cols, nchannels = 28, 28, 1
    nb_classes = 10

    print('Crafting ' + str(source_samples) + ' * ' + str(nb_classes - 1) +
        ' adversarial examples')

    # Keep track of success (adversarial example classified in target)
    results = np.zeros((nb_classes, source_samples), dtype='i')

    # Rate of perturbed features for each test set example and target class
    perturbations = np.zeros((nb_classes, source_samples), dtype='f')

    # Initialize our array for grid visualization
    grid_shape = (nb_classes, nb_classes, img_rows, img_cols, nchannels)
    grid_viz_data = np.zeros(grid_shape, dtype='f')

    # Instantiate a SaliencyMapMethod attack object
    wrap = KerasModelWrapper(loaded_model)
    jsma = SaliencyMapMethod(wrap, sess=sess)
    jsma_params = {'theta': 1., 'gamma': 0.1,
                   'clip_min': 0., 'clip_max': 1.,
                   'y_target': None}

    figure = None
    # Loop over the samples we want to perturb into adversarial examples
    for sample_ind in xrange(0, source_samples):
      print('--------------------------------------')
      print('Attacking input %i/%i' % (sample_ind + 1, source_samples))
      sample = X_test[sample_ind:(sample_ind + 1)]

      # We want to find an adversarial example for each possible target class
      # (i.e. all classes that differ from the label given in the dataset)
      current_class = int(np.argmax(y_test[sample_ind]))
      target_classes = other_classes(nb_classes, current_class)

      # For the grid visualization, keep original images along the diagonal
      grid_viz_data[current_class, current_class, :, :, :] = np.reshape(
          sample, (img_rows, img_cols, nchannels))

      # Loop over all target classes
      for target in target_classes:
        print('Generating adv. example for target class %i' % target)

        # This call runs the Jacobian-based saliency map approach
        one_hot_target = np.zeros((1, nb_classes), dtype=np.float32)
        one_hot_target[0, target] = 1
        jsma_params['y_target'] = one_hot_target
        adv_x = jsma.generate_np(sample, **jsma_params)

        # Check if success was achieved
        res = int(model_argmax(sess, x, predictions, adv_x) == target)

        # Computer number of modified features
        adv_x_reshape = adv_x.reshape(-1)
        test_in_reshape = X_test[sample_ind].reshape(-1)
        nb_changed = np.where(adv_x_reshape != test_in_reshape)[0].shape[0]
        percent_perturb = float(nb_changed) / adv_x.reshape(-1).shape[0]

        # Display the original and adversarial images side-by-side
        if viz_enabled:
          figure = pair_visual(
              np.reshape(sample, (img_rows, img_cols, nchannels)),
              np.reshape(adv_x, (img_rows, img_cols, nchannels)), figure)

        # Add our adversarial example to our grid data
        grid_viz_data[target, current_class, :, :, :] = np.reshape(
            adv_x, (img_rows, img_cols, nchannels))

        # Update the arrays for later analysis
        results[target, sample_ind] = res
        perturbations[target, sample_ind] = percent_perturb

    print('--------------------------------------')

    # Compute the number of adversarial examples that were successfully found
    nb_targets_tried = ((nb_classes - 1) * source_samples)
    succ_rate = float(np.sum(results)) / nb_targets_tried
    print('Avg. rate of successful adv. examples {0:.4f}'.format(succ_rate))
    report.clean_train_adv_eval = 1. - succ_rate

    # Compute the average distortion introduced by the algorithm
    percent_perturbed = np.mean(perturbations)
    print('Avg. rate of perturbed features {0:.4f}'.format(percent_perturbed))

    # Compute the average distortion introduced for successful samples only
    percent_perturb_succ = np.mean(perturbations * (results == 1))
    print('Avg. rate of perturbed features for successful '
          'adversarial examples {0:.4f}'.format(percent_perturb_succ))

    # Close TF session
    sess.close()

    # Finally, block & display a grid of all the adversarial examples
    if viz_enabled:
      import matplotlib.pyplot as plt
      plt.close(figure)
      _ = grid_visual(grid_viz_data)
        
      #adv_x = jsma(sess, x, predictions, 10, X_test, Y_test, 0, 0.5, 0, 1)
      #X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test], batch_size=128)
      #accuracy = model_eval(sess, x, y, predictions, X_test_adv, Y_test, args={ "batch_size" : 128 })
    '''
    sess.close()

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
