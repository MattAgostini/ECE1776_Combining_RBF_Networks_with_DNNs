#!/bin/bash 
for B in 0.01 0.1 1.0 2.0 3.0 5.0 10.0
do
    for I in {1..5}
    do
	python mnist_add_rbf.py mnist_mlp mnist_mlprbf_${B}_$I --betas $B > run_output/mnist_mlprbf_${B}_$I.txt 
    done
done

#for B in 0.01 0.1 1.0 2.0 3.0 5.0 10.0
#do
#    for I in `seq 1 30`
#    do
#	python mnist_add_rbf.py cnn_$I cnnrbf_${B}_$I --betas $B --cnn > cnnrbf_${B}_$I.txt
#    done
#done
