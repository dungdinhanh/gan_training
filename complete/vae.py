import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from complete.util import *

Normal = tf.contrib.distributions.Normal
Bernoulli = tf.contrib.distributions.Bernoulli


class Dense:
    def __init__(self, M1, M2, f=tf.nn.relu):
        self.W = tf.Variable(tf.random_normal(shape=(M1, M2)) * 2.0/np.sqrt(M2))
        self.b = tf.Variable(np.zeros(M2).astype(np.float32))
        self.f = f

    def forward(self, X):
        return self.f(tf.matmul(X, self.W) + self.b)


class VariationalAutoEncoder:

    def __init__(self, D, hidden_layers):
        self.X = tf.placeholder(tf.float32, shape=(None, D))

        # encoder
        self.encoders = []
        M_in = D
        for layer_size in hidden_layers[:-1]:
            h = Dense(M_in, layer_size)
            M_in = layer_size
            self.encoders.append(h)
        M = hidden_layers[-1]
        h = Dense(M_in, 2 * M, lambda x: x)
        self.encoders.append(h)

        # forward data in encoders:
        current_layers_values = self.X
        for hidden in self.encoders:
            current_layers_values = hidden.forward(current_layers_values)

        self.means = current_layers_values[:, :M]
        self.stddev = tf.nn.softplus(current_layers_values[:, M:]) + 1e-6

        # standard_normal = Normal(loc=self.means, scale=self.stddev)
        standard_normal = Normal(loc=np.zeros(M, dtype=np.float32), scale=np.ones(M, dtype=np.float32))
        e = standard_normal.sample(tf.shape(self.means)[0])
        self.Z = e * self.stddev + self.means
        # st = tf.contrib.bayesflow.stochastic_tensor
        # st = tf.contrib.bayesflow.stochastic_tensor
        # with st.value_type(st.SampleValue()):
        #     self.Z = st.StochasticTensor(Normal(loc=self.means, scale=self.stddev))


        # decoder
        self.decoders = []
        M_in = M
        for layer_size in reversed(hidden_layers[:-1]):
            h = Dense(M_in, layer_size)
            M_in = layer_size
            self.decoders.append(h)
        last_layer = Dense(M_in, D, lambda x: x)
        self.decoders.append(last_layer)

        # forward in decoder
        current_layers_values = self.Z
        for layer in self.decoders:
            current_layers_values = layer.forward(current_layers_values)

        logits = current_layers_values

        self.X_hat_distribution = Bernoulli(logits)
        self.posterior_predictive = self.X_hat_distribution.sample()
        self.posterior_predictive_probs = tf.nn.sigmoid(logits)


        #_______Prior
        standard_normal = Normal(loc=np.zeros(M, dtype=np.float32),
                                 scale=np.ones(M, dtype=np.float32))

        Z_std = standard_normal.sample(1)
        current_layers_values = Z_std
        #forward in decoders
        for layer in self.decoders:
            current_layers_values = layer.forward(current_layers_values)

        logits = current_layers_values
        prior_predictive_dist = Bernoulli(logits)
        self.prior_predictive = prior_predictive_dist.sample()
        self.prior_predictive_probs = tf.nn.sigmoid(logits)


        # prior predictive from input
        self.Z_input = tf.placeholder(tf.float32, shape=(None, M))
        current_layers_values = self.Z_input
        for layer in self.decoders:
            current_layers_values = layer.forward(current_layers_values)
        logits = current_layers_values
        self.prior_predictive_input_probs = tf.nn.sigmoid(logits)


        # cost
        # kl = tf.reduce_sum(tf.contrib.distributions.kl_divergence(self.Z.distribution, standard_normal), 1)
        kl = -tf.log(self.stddev) + 0.5 * (self.stddev ** 2 + self.means ** 2) - 0.5
        kl = tf.reduce_sum(kl, axis=1)
        expected_log_likelihood = tf.reduce_sum(self.X_hat_distribution.log_prob(self.X), 1)
        self.elbo = tf.reduce_sum(expected_log_likelihood - kl)
        self.train_op = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(-self.elbo)
        # session
        self.init_var = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init_var)

    def fit(self, X, epochs=30, batch_sz=64):
        n = len(X)
        cost = []
        n_batches = n//batch_sz
        print("number of batches :", n_batches)
        for i in range(epochs):
            print("epoch", i)
            np.random.shuffle(X)
            for j in range(n_batches):
                batch = X[j * batch_sz: (j+1) * batch_sz]
                _, c, = self.sess.run((self.train_op, self.elbo), feed_dict={self.X:batch})
                c /= batch_sz
                if j % 100 == 0:
                    print("iter %d: cost %.3f" %(j, c))
                cost.append(c)
        plt.plot(cost)
        plt.show()

    def posterior_sampling(self, X):
        return self.sess.run(self.posterior_predictive, feed_dict={self.X: X})

    def prior_sample_from_input(self, Z):
        return self.sess.run(self.prior_predictive_input_probs, feed_dict={self.Z_input: Z})

    def prior_sample(self):
        return self.sess.run((self.prior_predictive, self.prior_predictive_probs))


if __name__ == '__main__':
    X, Y = get_mnist()
    X = (X > 0.5).astype(np.float32)
    done = False
    vae = VariationalAutoEncoder(784, [200, 100])
    vae.fit(X)
    while not done:
        i = np.random.choice(len(X))
        x = X[i]

        im = vae.posterior_sampling([x]).reshape(28, 28)
        plt.subplot(1, 2, 1)
        plt.imshow(x.reshape(28, 28), cmap='gray')
        plt.title("Original")
        plt.subplot(1, 2, 2)
        plt.imshow(im, cmap='gray')
        plt.title("Sampled")
        plt.show()

        ans = input("Generate another?")
        if ans and ans[0] in ('n' or 'N'):
            done = True

    done = False
    while not done:
        im, probs = vae.prior_sample()
        plt.subplot(1, 2, 1)
        plt.imshow(im.reshape(28, 28), cmap='gray')
        plt.title("Prior predictive sample")
        plt.subplot(1, 2, 2)
        plt.imshow(probs.reshape(28, 28), cmap='gray')
        plt.title("Prior predictive probs")
        plt.show()

        ans = input("Generate another?")
        if ans and ans[0] in ('n' or 'N'):
            done = True










