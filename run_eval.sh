#for I in {1..5}
#do
#    python eval_adversarial.py mnist_mlp >> eval_output/mnist_mlp_adversarial.txt 
#done

#for B in 0.01 0.1 1.0 2.0 3.0 5.0 10.0
#do
#    for I in {1..5}
#    do
#	 python eval_adversarial.py mnist_mlprbf_${B}_$I --rbf >> eval_output/mnist_mlprbf_${B}_adversarial.txt 
#    done
#done

#for I in {1..5}
#do
#    python eval_adversarial.py mnist_cnn --cnn >> eval_output/mnist_cnn_adversarial.txt 
#done

#for B in 0.01 0.1 1.0 2.0 3.0 5.0 10.0
#do
#    for I in {1..5}
#    do
#	 python eval_adversarial.py mnist_cnnrbf_${B}_$I --rbf --cnn >> eval_output/mnist_cnnrbf_${B}_adversarial.txt 
#    done
#done