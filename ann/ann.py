import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weigths = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    out_put = tf.matmul(inputs, Weigths) + biases
    if activation_function is not None:
        out_put = activation_function(out_put)
    return out_put


x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.005, x_data.shape).astype(np.float32)
bias = 0.5
y_data = np.square(x_data) + bias + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])


layer1 = add_layer(xs, 1, 10, activation_function=tf.nn.softmax)
prediction = add_layer(layer1, 10, 1, activation_function=None)

loss = tf.losses.mean_squared_error(ys, prediction)
# loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
#                      reduction_indices=[1]))
# optimizer = tf.train.GradientDescentOptimizer(0.1)
optimizer = tf.train.AdamOptimizer(0.3)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:

    # for step in range(1, 1000):
    #     sess.run(train, feed_dict={xs: x_data, ys: y_data})
    #     if step % 10 == 0:
    #         print(step, ": ", sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

    sess.run(init)
    plt.figure(1)
    plt.ion()
    # sf1 = plt.subplot(211)
    # sf2 = plt.subplot(212)
    error = np.array([0])

    for step in range(10000):
        sess.run(train, feed_dict={xs: x_data, ys: y_data})
        if step % 10 == 0:

            plt.clf()

            plt.subplot(211)
            plt.scatter(x_data, y_data)   # input data
            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            plt.plot(x_data, prediction_value, 'r-')
            err = sess.run(loss, feed_dict={xs: x_data, ys: y_data})

            error = np.append(error, err)
            index = m = np.linspace(1, len(error), len(error), dtype=np.int)

            plt.subplot(212)
            plt.plot(index, error, 'b-')
            # print(error, "|")
            # print(index)

            plt.subplot(212)
            plt.pause(0.01)
    plt.ioff()
    plt.show()

