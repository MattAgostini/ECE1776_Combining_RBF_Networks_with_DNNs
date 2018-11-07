import sys
import os
import numpy as np
import matplotlib.pyplot as plt

betas = ('0.01', '0.1', '1.0', '2.0', '3.0', '5.0', '10.0')
y_pos = np.arange(len(betas))
testAccuracy = []
adversarialAccuracy = [] 

test, advers = [], []  
t_count, a_count = 0, 0

mnist_mlprbf_filenames = ('mnist_mlprbf_0.01', 'mnist_mlprbf_0.1', 'mnist_mlprbf_1.0', 'mnist_mlprbf_2.0', 'mnist_mlprbf_3.0', 'mnist_mlprbf_5.0', 'mnist_mlprbf_10.0', )

count = 0
for filename in mnist_mlprbf_filenames:
    if "mnist_mlprbf_" in filename:
        print (filename)
        file = open("eval_output/"+filename+"_adversarial.txt", 'r')
        
        for line in file:

            if not line.startswith("Test accuracy"):
                continue

            acc = float(line.split(":")[1])

            if line.startswith("Test accuracy on adversarial"):
                advers.append(100*acc)
                a_count += 1
                continue

            if line.startswith("Test accuracy on legitimate"):
                test.append(100*acc)
                t_count += 1
                continue
            
        assert t_count == a_count
        
        print(np.mean(test), np.std(test), np.min(test), np.max(test))
        print(np.mean(advers), np.std(advers), np.min(advers), np.max(advers))
        testAccuracy.append(np.mean(test))
        adversarialAccuracy.append(np.mean(advers))
        test, advers = [], []  
        t_count, a_count = 0, 0
        count = count + 1


plt.bar(betas, adversarialAccuracy, align='center')
plt.xticks(y_pos, betas)
plt.ylabel('Accuracy')
plt.ylim(0,100)
plt.title('Accuracy on Adversarial Samples')
#plt.show()
plt.savefig('fig/mnist_mlprbf_adversarial.png')

plt.bar(betas, testAccuracy, align='center')
plt.xticks(y_pos, betas)
plt.ylabel('Accuracy')
plt.ylim(0,100)
plt.title('Accuracy on Legitimate Samples')
#plt.show()
plt.savefig('fig/mnist_mlprbf_legitimate.png')




