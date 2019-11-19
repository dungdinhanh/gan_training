import numpy as np
from scipy.stats import multivariate_normal as mvn
from complete.util import *
from matplotlib import pyplot as plt
from sklearn.mixture import BayesianGaussianMixture

class SingleGauss:
    means_ = None
    count = None

    @staticmethod
    def get_mean():
        if SingleGauss.means_ is None:
            SingleGauss.means_ = []
            SingleGauss.count = 0
            return SingleGauss.means_
        else:
            return SingleGauss.means_

    def __init__(self):
        self.y = None
        self.mean = None
        self.cov = None

    def fit(self, X):
        self.mean = X.mean(axis=0)
        mean = SingleGauss.get_mean()
        mean.append(self.mean)
        self.y = SingleGauss.count
        SingleGauss.count+=1
        self.cov = np.cov(X.T)

    def sample(self):
        return np.array([mvn.rvs(mean=self.mean, cov=self.cov)]), self.y


class BayesianSampler:
    def __init__(self, model):
        self.model = model

    def fit(self, X, Y, fit_clusters=False ,default_clusters=10):
        self.K = len(set(Y))
        self.gaussians = []
        self.mean_y = Y.mean()
        self.std_y = Y.std()
        for k in range(self.K):
            print("Fitting model " + str(self.model), k)
            Xk = X[Y==k]
            if not fit_clusters:
                mod = self.model()
            else:
                mod = self.model(default_clusters)
            mod.fit(Xk)
            self.gaussians.append(mod)

    def sample_given_y(self, y: int):
        mod = self.gaussians[y]
        sample = mod.sample()
        mean = mod.means_[sample[1]]
        return sample[0].reshape(28, 28), mean.reshape(28, 28)
        pass

    def sample(self):
        y = max(0, min(10, np.random.normal(self.mean_y, self.std_y)))
        return self.sample_given_y(int(y))


if __name__ == '__main__':
    # b = BayesianSampler(BayesianGaussianMixture)
    # mnist = get_mnist()
    # b.fit(mnist[0], mnist[1], fit_clusters=True)
    #
    # for k in range(b.K):
    #     gen, mean = b.sample_given_y(k)
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(gen, cmap='gray')
    #     plt.title("generatate")
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(mean, cmap='gray')
    #     plt.title("mean")
    #     plt.show()
    #
    # gen, mean = b.sample()
    # plt.subplot(1, 2, 1)
    # plt.imshow(gen, cmap='gray')
    # plt.title("random generate")
    # plt.subplot(1, 2, 2)
    # plt.imshow(gen, cmap='gray')
    # plt.title("random mean")
    # plt.show()

    b = BayesianSampler(SingleGauss)
    mnist = get_mnist()
    b.fit(mnist[0], mnist[1])
    k = input("Input digit: ")

    gen, mean = b.sample_given_y(int(k))
    plt.subplot(1, 2, 1)
    plt.imshow(gen, cmap='gray')
    plt.title("generatate")
    plt.subplot(1, 2, 2)
    plt.imshow(mean, cmap='gray')
    plt.title("mean")
    plt.show()


    for k in range(b.K):
        gen, mean = b.sample_given_y(k)
        plt.subplot(1, 2, 1)
        plt.imshow(gen, cmap='gray')
        plt.title("generatate")
        plt.subplot(1, 2, 2)
        plt.imshow(mean, cmap='gray')
        plt.title("mean")
        plt.show()

    gen, mean = b.sample()
    plt.subplot(1, 2, 1)
    plt.imshow(gen, cmap='gray')
    plt.title("random generate")
    plt.subplot(1, 2, 2)
    plt.imshow(gen, cmap='gray')
    plt.title("random mean")
    plt.show()

