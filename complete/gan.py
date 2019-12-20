import numpy as np
import tensorflow as tf
import os
from util import *
from datetime import *
import scipy as sp
from matplotlib import pyplot as plt
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


LEARNING_RATE = 0.0002
BETA1 = 0.5
BATCH_SIZE = 64
EPOCHS = 2
SAVE_SAMPLE_PERIOD = 50

DECAY=0.9
EPS=1e-5
D_SCOPE="discriminator"
G_SCOPE="generator"

CONV = "conv"
DENSE = "dense"

if not os.path.exists("samples"):
    os.makedirs("samples")


def lrelu(x, alpha=0.2):
    return tf.maximum(alpha * x, x)


class ConvLayer:
    def __init__(self, name, m_in, m_out, apply_batch_norm, filter_sz=5, strides=2, f=tf.nn.relu):
        self.name = name
        self.m_in = m_in
        self.m_out = m_out
        self.apply_batch_norm = apply_batch_norm
        self.filter_sz = 5
        self.strides = strides
        self.f = f
        self.W = tf.get_variable("W_%s"%name, shape=(filter_sz, filter_sz, m_in, m_out),
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.b = tf.get_variable("b_%s"%name, shape=(m_out,),
                                 initializer=tf.zeros_initializer)
        self.params = [self.W, self.b]
        pass

    def forward(self, X, reuse, is_training):
        conv_out = tf.nn.conv2d(
            X,
            self.W,
            strides=[1, self.strides, self.strides, 1],
            padding='SAME'
        )

        conv_out = tf.nn.bias_add(conv_out, self.b)

        if self.apply_batch_norm:
            conv_out = tf.contrib.layers.batch_norm(
                conv_out,
                decay=0.9,
                updates_collections=None,
                epsilon=1e-5,
                scale=True,
                is_training=is_training,
                reuse=reuse,
                scope=self.name
            )
        return self.f(conv_out)


class FractionallyStrideConvLayer:
    def __init__(self, name, m_in, m_out, output_shape, apply_batch_norm, filtersz=5, stride=2, f=tf.nn.relu):
        self.W = tf.get_variable(
            "W_%s"%name,
            shape=(filtersz, filtersz, m_out, m_in),
            initializer=tf.random_normal_initializer(stddev=0.02),
        )

        self.b = tf.get_variable(
            "b_%s"%name,
            shape=(m_out,)
        )
        self.f = f
        self.stride = stride
        self.name = name
        self.output_shape = output_shape
        self.apply_batch_norm = apply_batch_norm
        self.params = [self.W, self.b]

    def forward(self, X, reuse, is_training):
        conv_out = tf.nn.conv2d_transpose(
            value=X,
            filter=self.W,
            output_shape=self.output_shape,
            strides=[1, self.stride, self.stride, 1],
        )

        conv_out = tf.nn.bias_add(conv_out, self.b)

        # apply batch normalization
        if self.apply_batch_norm:
            conv_out = tf.contrib.layers.batch_norm(
                conv_out,
                decay=DECAY,
                updates_collections=None,
                epsilon=EPS,
                scale=True,
                is_training=is_training,
                reuse=reuse,
                scope=self.name,
            )
        return self.f(conv_out)

class DenseLayer(object):
    def __init__(self, name, M1, M2, apply_batch_norm, f=tf.nn.relu):
        self.W = tf.get_variable(
            "W_%s"%name,
            shape=(M1, M2),
            initializer=tf.random_normal_initializer(stddev=0.02),
        )

        self.b=tf.get_variable(
            "b_%s"%name,
            shape=(M2,),
            initializer=tf.zeros_initializer(),
        )

        self.f = f
        self.apply_batch_norm = apply_batch_norm
        self.M1 = M1
        self.M2 = M2
        self.name = name

    def forward(self, X, reuse, is_training):
        a = tf.matmul(X, self.W) + self.b

        if self.apply_batch_norm:
            a = tf.contrib.layers.batch_norm(
                a,
                decay=DECAY,
                updates_collections=None,
                epsilon=EPS,
                scale=True,
                is_training=is_training,
                reuse=reuse,
                scope=self.name
            )
        return self.f(a)


class Discriminator:
    def __init__(self, dim, colors, d_sizes):
        self.d_sizes = d_sizes
        self.dim = dim
        self.colors = colors
        self.build()
        pass

    def build(self):
        with tf.variable_scope(D_SCOPE) as scope:
            count = 0
            mi = self.colors
            self.d_conv = []
            out_dim = self.dim
            mo = mi
            for size in self.d_sizes[CONV]:
                name = "d_" + CONV + "_%s" % count
                count += 1
                mo = size[0]
                stride = size[2]
                layer = ConvLayer(name, mi, mo, size[-1], size[1], stride, lrelu)
                out_dim = int(np.ceil(float(out_dim) / stride))
                self.d_conv.append(layer)
                mi = mo

            self.d_dense = []
            mi = mo * out_dim * out_dim
            count = 0
            for size in self.d_sizes[DENSE]:
                name = "d_" + DENSE + "_%s" % count
                mo = size[0]
                layer = DenseLayer(name, mi, size[0], size[1], lrelu)
                self.d_dense.append(layer)
                mi = mo
                count += 1

            # final dense
            name = "d_" + DENSE + "_%s" % count
            self.d_final_dense = DenseLayer(name, mi, 1, False, lambda x: x)
            return self.d_conv, self.d_dense, self.d_final_dense

    def forward(self, X, reuse=None, is_training=True):
        out = X
        for layer in self.d_conv:
            out = layer.forward(out, reuse, is_training)
        out = tf.contrib.layers.flatten(out)
        for layer in self.d_dense:
            out = layer.forward(out, reuse, is_training)
        logits = self.d_final_dense.forward(out, reuse, is_training)
        return logits


class Generator(object):
    def __init__(self, dim, colors, batch_sz, g_sizes):
        self.dim = dim
        self.colors = colors
        self.g_sizes = g_sizes
        self.batch_sz = batch_sz
        self.build()

    def build(self):
        with tf.variable_scope(G_SCOPE) as scope:
            self.latent_dims = self.g_sizes['z']
            self.dims = [self.dim]
            dim = self.dim
            for layer in reversed(self.g_sizes[CONV]):
                dim = int(np.ceil(float(dim)/layer[2]))
                self.dims.append(dim)

            self.dims = list(reversed(self.dims))

            mi = self.latent_dims
            count = 0

            self.g_dense = []
            for size in self.g_sizes[DENSE]:
                mo = size[0]
                name = "g_" + DENSE + "_%s"%count
                layer = DenseLayer(name, mi, mo, size[1])
                self.g_dense.append(layer)
                count += 1
                mi = mo

            mo = self.g_sizes['projection'] * self.dims[0] * self.dims[0]
            name = "g_" + DENSE + "_%s"%count
            self.g_final_dense = DenseLayer(name, mi, mo, not self.g_sizes['bn_after_project'])

            self.g_conv = []
            count = 0
            mi = self.g_sizes['projection']

            #output may use tanh or sigmoid
            num_relus = len(self.g_sizes[CONV]) - 1
            activation_functions = [tf.nn.relu] * num_relus + [self.g_sizes['output_activation']]
            for size in self.g_sizes[CONV]:
                name = "fs_g_" + CONV + "_%s"%count
                mo = size[0]
                filter_size = size[1]
                stride = size[2]
                batch_norm = size[3]
                output_shape = [self.batch_sz, self.dims[count+1], self.dims[count+1], mo]
                layer = FractionallyStrideConvLayer(name, mi, mo, output_shape, batch_norm, filter_size,
                                                    stride, activation_functions[count])
                self.g_conv.append(layer)
                count += 1
                mi = mo

            return self.g_conv, self.g_dense, self.g_final_dense

    def forward(self, X, reuse=None, is_training=True):
        out = X
        for layer in self.g_dense:
            out = layer.forward(out, reuse, is_training)

        # project and reshape
        out = self.g_final_dense.forward(out, reuse, is_training)
        out = tf.reshape(out, [-1, self.dims[0], self.dims[0], self.g_sizes['projection']])

        # apply batch norm
        if self.g_sizes['bn_after_project']:
            out = tf.contrib.layers.batch_norm(
                out,
                decay=DECAY,
                updates_collections=None,
                epsilon=1e-5,
                scale=True,
                is_training = is_training,
                reuse = reuse,
                scope='bn_after_project'
            )

        for layer in self.g_conv:
            out = layer.forward(out, reuse, is_training)
        return out


class DCGAN(object):
    def __init__(self, dim, colors, g_sizes, d_sizes):
        # dim is the size of the photo
        # colors is the number of colors in photo
        self.dim = dim
        self.colors = colors
        self.g_sizes = g_sizes
        self.d_sizes = d_sizes
        self.latent_dims = g_sizes['z']




        self.X = tf.placeholder(
            tf.float32,
            shape=(None, dim, dim, colors),
            name='X'
        )

        self.Z = tf.placeholder(
            tf.float32,
            shape=(None, g_sizes['z']),
            name='z'
        )


        self.batch_sz = tf.placeholder(tf.int32, shape=(), name='batch_sz')

        generator = Generator(dim, colors, self.batch_sz,g_sizes)
        discriminator = Discriminator(dim, colors, d_sizes)

        with tf.variable_scope(D_SCOPE) as scope:
            # scope.reuse_variables()
            logits = discriminator.forward(self.X)

        with tf.variable_scope(G_SCOPE) as scope:
            # scope.reuse_variables()
            self.sample_images = generator.forward(self.Z)

        with tf.variable_scope(D_SCOPE) as scope:
            scope.reuse_variables()
            sample_logits = discriminator.forward(self.sample_images, True)

        with tf.variable_scope(G_SCOPE) as scope:
            scope.reuse_variables()
            self.test_samples = generator.forward(self.Z, True, False)

        # build cost

        self.d_cost_real = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits,
            labels=tf.ones_like(logits)
        )

        self.d_cost_fake = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=sample_logits,
            labels=tf.zeros_like(sample_logits)
        )


        self.d_cost = tf.reduce_mean(self.d_cost_real) + tf.reduce_mean(self.d_cost_fake)
        self.g_cost = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=sample_logits,
                labels=tf.ones_like(sample_logits)
            )
        )

        real_predictions = tf.cast(logits > 0, tf.float32)
        fake_predictions = tf.cast(sample_logits < 0, tf.float32)
        num_predictions = 2.0 * BATCH_SIZE
        num_correct = tf.reduce_sum(real_predictions) + tf.reduce_sum(fake_predictions)
        self.d_accuracy = num_correct/num_predictions * 100

        # optimizers
        self.d_params = [t for t in tf.trainable_variables() if t.name.startswith('d')]
        self.g_params = [t for t in tf.trainable_variables() if t.name.startswith('g')]

        self.d_train_op = tf.train.AdamOptimizer(
            LEARNING_RATE, beta1=BETA1
        ).minimize(
            self.d_cost, var_list=self.d_params
        )

        self.g_train_op = tf.train.AdamOptimizer(
            LEARNING_RATE, beta1=BETA1
        ).minimize(
            self.g_cost, var_list=self.g_params
        )


        # set up session
        self.init_op = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init_op)
        pass

    def fit(self, X):
        d_costs = []
        g_costs = []

        N = len(X)
        n_batches = N // BATCH_SIZE
        total_iters = 0
        for i in range(EPOCHS):
            print("epoch:", i)
            np.random.shuffle(X)
            for j in range(n_batches):
                t0 = datetime.now()
                if type(X[0]) is str:
                    pass
                else:
                    # is mnist
                    batch = X[j * BATCH_SIZE: (j + 1) * BATCH_SIZE]
                Z = np.random.uniform(-1, 1, size=(BATCH_SIZE, self.latent_dims))
                # train discriminator
                _, d_cost, d_acc = self.sess.run(
                    (self.d_train_op, self.d_cost, self.d_accuracy),
                    feed_dict={self.X: batch, self.Z: Z, self.batch_sz: BATCH_SIZE}
                )

                d_costs.append(d_cost)

                # train generator
                _, g_cost1 = self.sess.run(
                    (self.g_train_op, self.g_cost),
                    feed_dict={self.Z: Z, self.batch_sz: BATCH_SIZE}
                )


                _, g_cost2 = self.sess.run(
                    (self.g_train_op, self.g_cost),
                    feed_dict={self.Z: Z, self.batch_sz:BATCH_SIZE}
                )

                g_costs.append((g_cost1 + g_cost2)/2)
                print (" batch: %d/%d - dt: %s - d_acc: %.2f"%(j + 1, n_batches, datetime.now() - t0, d_acc))


                # save sampels periodically
                total_iters += 1
                d = self.dim
                if total_iters % SAVE_SAMPLE_PERIOD == 0:
                    print("saving a sample...")
                    samples = self.sample(64)

                    # for convenience
                    # d = self.dim

                    if samples.shape[-1] == 1:
                        samples = samples.reshape(64, d, d)
                        flat_image = to_images(samples, d, 8, 8)
                    else:
                        flat_image = np.empty((8*d, 8 *d))
                        pass
                    sp.misc.imsave(
                        'samples/samples_at_iter_%d.png' %total_iters,
                        flat_image
                    )
        plt.clf()
        plt.plot(d_costs, label='discriminator cost')
        plt.plot(g_costs, label='generator cost')
        plt.legend()
        plt.savefig('cost_vs_iteration.png')


    def sample(self, n):
        Z = np.random.uniform(-1, 1, size=(n, self.latent_dims))
        samples = self.sess.run(self.test_samples, feed_dict={self.Z: Z, self.batch_sz: n})

        return samples


def to_images(samples, dim,n, m):
    flat_image = np.empty((n * dim, m * dim))
    k = 0

    for i in range(8):
        for j in range(8):
            flat_image[i * dim: (i + 1) * dim, j*dim: (j+1) * dim] = samples[k].reshape(dim, dim)
            k+=1
        plt.imshow(flat_image, cmap='gray')
    return flat_image


def mnist():
    X, Y = get_mnist()
    X = X.reshape(len(X), 28, 28, 1)
    dim = X.shape[1]
    colors = X.shape[-1]

    devices = get_available_gpus()



    d_sizes = {
        CONV: [(2, 5, 2, False), (64, 5, 2, True)],
        DENSE: [(1024, True)]
    }

    g_sizes = {
        'z': 100,
        'projection': 128,
        'bn_after_project': False,
        CONV:[(128, 5, 2, True), (colors, 5, 2, False)],
        DENSE: [(1024, True)],
        'output_activation': tf.sigmoid
    }
    if len(devices) != 0:
        with tf.device("/gpu:0"):
            gan = DCGAN(dim, colors, g_sizes, d_sizes)
        pass
    else:
        gan = DCGAN(dim, colors,g_sizes, d_sizes )
    gan.fit(X)

if __name__ == '__main__':
    mnist()

