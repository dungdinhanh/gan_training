from peripheral.util import *
from exercises.bayes_sampling import *
import png
from PIL import Image
from code_sample.bayes_classifier_gaussian import *
import matplotlib.pyplot as plt

mnist = get_mnist()
b = Bayesian(mnist[0], mnist[1])
b.bayes_sampler_caculate(7)
c = b.sample()
c = c.reshape((28, 28))
# print(c)
# png.from_array(c, 'L').save("abc.png")
print(c)

binary_transform = np.array(c).astype(np.uint8)
# binary_transform[binary_transform > 0] = 255
img = Image.fromarray(binary_transform, 'P')
img.save("tucode_co_y.png")

d = b.sample_2()
d = d.reshape((28,28))
print(d)
binary_transform = np.array(d).astype(np.uint8)
img2 = Image.fromarray(binary_transform, 'P')
img2.save("tucode_khong_y.png")


X, Y = mnist[0], mnist[1]
clf = BayesClassifier()
clf.fit(X, Y)
for k in range(clf.K):
    sample = clf.sample_given_y(k).reshape(28,28)
    mean = clf.gaussians[k]['m'].reshape(28,28)

    plt.subplot(1, 2, 1)
    plt.imshow(sample, cmap='gray')
    plt.title("Sample")
    plt.subplot(1, 2, 2)
    plt.imshow(mean, cmap='gray')
    plt.title("Mean")
    plt.show()

# generate a random sample
sample = clf.sample().reshape(28,28)
plt.imshow(sample, cmap='gray')
plt.title("Random Sample From Random Class")
plt.show()


