from sklearn.mixture import BayesianGaussianMixture
from peripheral.util import *
from matplotlib import pyplot as plt
from code_sample.bayes_classifier_gaussian import *


bgm = BayesMixClassifier()
mnist = get_mnist()
bgm.fit(mnist[0], mnist[1])
sample = bgm.sample()

plt.subplot(1, 2, 1)
plt.imshow(sample[0], cmap='gray')
plt.title("mixture ")
plt.show()

plt.subplot(1, 2, 2)
plt.imshow(sample[1], cmap='gray')
plt.title("mean")
plt.show()