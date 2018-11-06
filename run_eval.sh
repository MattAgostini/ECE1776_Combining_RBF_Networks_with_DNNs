#for I in {1..5}
#do
#    python eval_adversarial.py mnist_mlp_$I >> mnist_mlp_adversarial.txt 
#done

for B in 0.01 0.1 1.0 2.0 3.0 5.0 10.0
do
    for I in {1..5}
    do
	python eval_adversarial.py mnist_mlprbf_${B}_$I --rbf >> eval_output/mnist_mlprbf_${B}_adversarial.txt 
    done
done
