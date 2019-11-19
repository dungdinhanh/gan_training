import numpy as np
from scipy.stats import multivariate_normal as mvn
from sklearn.mixture import BayesianGaussianMixture
import random

class BayesClassifier:
    def fit(self, X, Y):
        self.K = len(set(Y))
        self.gaussians = []


        for k in range(self.K):
            Xk = X[Y==k]
            mean = Xk.mean(axis=0)
            cov = np.cov(Xk.T)
            g = {'m': mean, 'c': cov}
            self.gaussians.append(g)

    def sample_given_y(self, y):
        g = self.gaussians[y]
        return mvn.rvs(mean=g['m'], cov=g['c'])

    def sample(self):
        y = np.random.randint(self.K)
        return self.sample_given_y(y)


class BayesMixClassifier:
    def fit(self, X, Y):
        self.K = len(set(Y))
        self.gaussians = []

        for k in range(self.K):
            print("Fitting gmm ", k)
            Xk = X[Y==k]
            gmm = BayesianGaussianMixture(10)
            gmm.fit(Xk)
            self.gaussians.append(gmm)


    def sample_given_y(self, y):
        gmm = self.gaussians[y]
        sample = gmm.sample()

        mean = gmm.means_[sample[1]]
        return sample[0].reshape(28, 28), mean.reshape(28,28)

    def sample(self):
        y = np.random.randint(self.K)
        return self.sample_given_y(y)



