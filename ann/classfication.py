import tensorflow as tf
import numpy as np
from lib import nn_build_helper

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

input_dim = 28 * 28
# print(batch_xs)
# exit()
xs = tf.placeholder(tf.float32, [None, input_dim])
ys = tf.placeholder(tf.float32, [None, 10])

hidden_layer1 = nn_build_helper.add_layer(xs, input_dim, 300, activation_function=tf.nn.relu, layer_name="hidden_layer_1")
# hidden_layer2 = common.add_layer(hidden_layer1, 300, 800, activation_function=tf.nn.relu, layer_name="hidden_layer_2")
# hidden_layer3 = common.add_layer(hidden_layer2, 800, 2048, activation_function=tf.nn.relu, layer_name="hidden_layer_3")
prediction = nn_build_helper.add_layer(hidden_layer1, 300, 10, activation_function=tf.nn.softmax, layer_name="prediction")
# loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))


# cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=ys)
# loss = tf.losses.softmax_cross_entropy(logits=prediction, onehot_labels=ys)
# loss = tf.losses.mean_squared_error(predictions=prediction, labels=ys)
# loss = tf.losses.mean_squared_error(predictions=prediction, labels=ys)
# loss = tf.losses.sparse_softmax_cross_entropy(logits=prediction, labels=ys)


y_clipped = tf.clip_by_value(prediction, 1e-10, 0.9999999)
loss = -tf.reduce_mean(tf.reduce_sum(ys * tf.log(y_clipped)
                                                  + (1 - ys) * tf.log(1 - y_clipped), axis=1))

train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for step in range(1000000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train, feed_dict={xs: batch_xs, ys: batch_ys})
        if step % 50 == 0:
            print(nn_build_helper.compute_accuracy(mnist.test.images, mnist.test.labels, sess, prediction, xs))
