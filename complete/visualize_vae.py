from complete.util import *
from complete.vae  import VariationalAutoEncoder
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

#
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)


if __name__ == '__main__':
    # with tf.device("/gpu:0"):
    X, Y = get_mnist()
    X = (X > 0.5).astype(np.float32)

    vae = VariationalAutoEncoder(784, [200, 100, 2])
    vae.fit(X)

    n = 20
    x_space = np.linspace(-10, 10, n)
    y_space = np.linspace(-10, 10, n)
    image = np.empty((28 * n, 28 * n))
    Z = []
    for i, x in enumerate(x_space):
        for j, y in enumerate(y_space):
            z = [x, y]
            Z.append(z)
    X_recon = vae.prior_sample_from_input(Z)
    k = 0
    for i, x in enumerate(x_space):
        for j, y in enumerate(y_space):
            x_recon = X_recon[k]
            k+=1
            x_recon = x_recon.reshape((28, 28))
            image[i * 28: (i + 1) * 28, j*28: (j+1) * 28] = x_recon
    plt.imshow(image, cmap='gray')
    plt.show()

