import numpy as np



class Bayesian:

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.mean = None
        self.covariance = None
        self.std = None

        self.mean_y = None
        self.std_y = None
        self.covariance_y = None

    def get_indexes_for_y(self, X: np.ndarray ,Y: np.ndarray, y: int):
        indexes = np.where(Y == y)
        return X[indexes]

    def calculate_p_y(self):
        self.mean_y = np.mean(self.Y)
        self.std_y = np.std(self.Y)
        self.covariance_y = np.cov(self.Y)

    def bayes_sampler_caculate(self, y: int):
        if self.X is None or self.Y is None:
            return None
        self.mean = np.zeros(self.X.shape[1])
        self.covariance = np.zeros(self.X.shape[1])
        self.std = np.zeros(self.X.shape[1])
        X_y = self.get_indexes_for_y(self.X, self.Y, y)
        X_y_t = np.transpose(X_y)
        n = X_y_t.shape[0]
        for i in range(n):
            self.mean[i] = np.mean(X_y_t[i])
            self.std[i] = np.std(X_y_t[i])
            self.covariance[i] = np.cov(X_y_t[i])
        pass


    def sample(self):
        if self.mean is None or self.covariance is None or self.std is None:
            return None
        n = self.mean.shape[0]
        data = np.zeros([n,])
        for i in range(n):
            data[i] = min(255, max(0, np.random.normal(self.mean[i], self.std[i])))

        data = np.round(data)
        return data

    def sample_2(self):
        self.calculate_p_y()
        y = min(9, max(0, np.random.normal(self.mean_y, self.std_y)))
        y = np.round([y])[0]
        self.bayes_sampler_caculate(y)
        return self.sample()








