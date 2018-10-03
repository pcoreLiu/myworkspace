# a simple demo for Perceptron
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return sess.run(Weights)*x + sess.run(biases)


x_data = np.random.rand(20).astype(np.float32)
y_data = x_data * 2.9 + 0.3

Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

loss = tf.losses.mean_squared_error(y_data, y)
optimizer = tf.train.GradientDescentOptimizer(0.1)

# loss = tf.reduce_mean(tf.square(y - y_data))
# optimizer = tf.train.GradientDescentOptimizer(0.1)
# optimizer = tf.train.AdamOptimizer(0.9)

train = optimizer.minimize(loss)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    plt.figure(1)
    plt.ion()
    # sf1 = plt.subplot(211)
    # sf2 = plt.subplot(212)
    error = np.array([0])

    for step in range(1000):
        sess.run([train, loss])
        if step % 10 == 0:
            print(step, sess.run(Weights), sess.run(biases))
            plt.clf()

            plt.subplot(211)
            plt.scatter(x_data, y_data)
            plt.plot(x_data, f(x_data), 'r-')

            error = np.append(error, loss.eval())
            index = np.array([0])
            for i in range(1, len(error)):
                index = np.append(index, i)

            plt.subplot(212)
            plt.plot(index, error, 'b-')
            # print(error, "|")
            # print(index)

            plt.subplot(212)
            plt.pause(0.01)
    plt.ioff()
    plt.show()





















