# ECE1776_Combining_RBF_Networks_with_DNNs

'models' directory contains models of differing architectures trained on various datasets before adding RBF layer.

'rbfmodels' directory holds the same pretrained architectures from 'models' but with an rbf layer appended at the end.

To generate models and rbfmodels:

1. Run one of the model generation scripts. This will store the trained model in 'models.'
train_models_mnist - Creates an MLP and CNN trained on the MNIST dataset
train_models_cifar10 - Creates a CNN trained on the CIFAR-10 dataset

2. Run mnist_add_rbf

This adds an RBF layer to a pretrained input model and trained the stacked architecture for 3 epochs.

Usage: mnist_add_rbf --betas [B] input_model output_model

input - input model saved in 'models' directory
output - output model to be saved in 'rbfmodels' directory

3. Run eval_adversarial

This will run an FGSM attack on an input network.

Usage: eval_adversarial input_model [--cnn] [--rbf]

Notes: 
Use --rbf if the input network contains an RBF layer
Use --cnn if the input network is a CNN

Scripts:
run.sh - Currently trains multiple stacked RBF networks with various Beta initializations. Output goes to run_output directory
run_eval.sh - Currently tests trained models on a given adversarial attack (FSGM at the moment). Output goes to eval_output directory
stat.sh - Currently in progress.........

