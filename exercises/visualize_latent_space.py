from peripheral import util
import numpy as np
import matplotlib.pyplot as plt
from exercises.vae import VariationalAutoencoder


if __name__ == '__main__':
    X, Y = util.get_mnist()

    X = ( X> 0.5).astype(np.float32)

    vae = VariationalAutoencoder(784, [200, 100, 2])
    vae.fit(X)
    n = 20

    x_values = np.linspace(-3, 3, n)
    y_values = np.linspace(-3, 3, n)
    image = np.empty((28*n, 28*n))

    Z = []
    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            z = [x, y]
            Z.append(z)
    X_recon = vae.prior_predictive_with_input(Z)

    k = 0
    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            x_recon = X_recon[k]
            k+=1
            x_recon = x_recon.reshape(28, 28)
            image[(n-i-1) * 28: (n-i)*28, j * 28: (j+1) * 28] = x_recon
    plt.imshow(image, cmap='gray')
    plt.show()
