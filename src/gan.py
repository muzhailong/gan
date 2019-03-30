import tensorflow as tf
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

in_dim = 28 * 28
h_dim = 128
g_dim = 100


def xaiver(size):
    in_dim = size[0]
    var = 1.0 / tf.sqrt(in_dim / 2.0)
    return tf.random_normal(shape=size, stddev=var)


X = tf.placeholder(dtype=tf.float32, shape=(None, in_dim))

D_w1 = tf.Variable(initial_value=xaiver((in_dim, h_dim)))
D_b1 = tf.Variable(initial_value=tf.zeros(shape=(h_dim,)))

D_w2 = tf.Variable(initial_value=xaiver(size=(h_dim, 1)))
D_b2 = tf.Variable(initial_value=tf.zeros(shape=[1, ]))

D_theta = [D_w1, D_b1, D_w2, D_b2]

Z = tf.placeholder(dtype=tf.float32, shape=(None, g_dim))

G_w1 = tf.Variable(initial_value=xaiver(size=(g_dim, h_dim)))
G_b1 = tf.Variable(initial_value=tf.zeros(shape=[h_dim, ]))

G_w2 = tf.Variable(initial_value=xaiver(size=(h_dim, in_dim)))
G_b2 = tf.Variable(initial_value=tf.zeros(shape=(in_dim,)))

G_theta = [G_w1, G_b1, G_w2, G_b2]


def generator(z):
    h = tf.nn.relu(tf.matmul(z, G_w1) + G_b1)
    G_logit = tf.matmul(h, G_w2) + G_b2
    G_prob = tf.nn.sigmoid(G_logit)
    return G_prob


def discriminator(x):
    h = tf.nn.relu(tf.matmul(x, D_w1) + D_b1)
    D_logit = tf.matmul(h, D_w2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)
    return D_logit, D_prob


G_sample = generator(Z)

D_logit_real, D_prob_real = discriminator(X)
D_logit_fake, D_prob_fake = discriminator(G_sample)

D_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake))
)

D_loss = D_loss_fake + D_loss_real

G_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=D_theta)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=G_theta)

epochs = 10
batch_size = 32

D_loss_arr = []
G_loss_arr = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    (x_train, y_train), _ = mnist.load_data()
    x_train = np.reshape(x_train, (-1, 28 * 28))
    x_train = x_train / 255
    for i in range(epochs):
        for j in range(int(len(x_train) / batch_size)):
            rand_index = np.random.choice(len(x_train), batch_size)
            _, _d_loss = sess.run([D_solver, D_loss],
                                  feed_dict={
                                      X: x_train[rand_index],
                                      Z: np.random.standard_normal(size=(len(rand_index), g_dim))
                                  })
            _, _g_loss = sess.run(fetches=[G_solver, G_loss], feed_dict={
                Z: np.random.standard_normal(size=(g_dim, g_dim))
            })
            D_loss_arr.append(_d_loss)
            G_loss_arr.append(_g_loss)

        # test
        samples = sess.run(G_sample, feed_dict={
            Z: np.random.standard_normal(size=(10, g_dim))
        })

        samples = np.round(samples * 255)
        samples = np.reshape(samples, (-1, 28, 28, 1))
        samples = samples.astype(np.uint8)
        if not os.path.exists("../reuslt/gan/" + str(i)):
            os.mkdir("../result/gan/" + str(i))
        for j in range(len(samples)):
            cv2.imwrite("../result/gan/" + str(i) + "/" + str(j) + ".jpg", samples[j])

    plt.plot(np.arange(len(D_loss_arr)), D_loss_arr, c="r")
    plt.plot(np.arange(len(G_loss_arr)), G_loss_arr, c="b")
    plt.legend(["discriminator", "generator"])
    plt.savefig("../result/gan/1.png")
    plt.show()
