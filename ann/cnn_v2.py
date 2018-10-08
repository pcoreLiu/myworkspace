import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from lib import common


def create_conv_layer(input_data, num_input_channels,
                      num_filters, filter_shape, filter_strides=[1, 1], padding="SAME", name=""):
    conv_filter_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]
    weights = tf.Variable(tf.truncated_normal(conv_filter_shape, stddev=0.03), name=name + "_W")
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name + "_b")

    out_layer = tf.nn.conv2d(input_data, weights, [1, filter_strides[0], filter_strides[1], 1], padding=padding)
    conv_layer = tf.nn.relu(tf.add(out_layer, bias))
    return conv_layer


def create_max_pool(conv_layer, pool_shape, pool_strides=[2, 2], padding="SAME", pool_name=""):
    kernel_size = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, pool_strides[0], pool_strides[1], 1]
    out_layer = tf.nn.max_pool(conv_layer, ksize=kernel_size, strides=strides, padding=padding,
                               name=pool_name + "maxPool")
    return out_layer


def run_cnn():
    mnist = input_data.read_data_sets("./MNIST_data", one_hot=True,
                                      source_url='http://storage.googleapis.com/cvdf-datasets/mnist/')

    learning_rate = 0.0001
    epochs = 50
    batch_size = 100

    input_dim = 28 * 28
    x = tf.placeholder(tf.float32, [None, input_dim])
    y = tf.placeholder(tf.float32, [None, 10])

    x_shaped = tf.reshape(x, [-1, 28, 28, 1])

    conv_1 = create_conv_layer(x_shaped, 1, 32, [5, 5], name="conv_layer_1")  # output: 28*28*32
    conv_1 = create_max_pool(conv_1, [2, 2])  # output: 14*14*32
    conv_2 = create_conv_layer(conv_1, 32, 64, [5, 5], name="conv_layer_2")  # output: 14*14*64
    conv_2 = create_max_pool(conv_2, [2, 2])  # output: 7*7*64

    flattened = tf.reshape(conv_2, [-1, 7 * 7 * 64])

    dense_layer_1 = common.add_layer(flattened, flattened.shape[1].value, 1568,
                                     activation_function=tf.nn.relu, layer_name="dense_layer_1")
    dense_layer_2 = common.add_layer(dense_layer_1, 1568, 392,
                                     activation_function=tf.nn.relu, layer_name="dense_layer_2")
    prediction = common.add_layer(dense_layer_2, 392, 10,
                                  activation_function=tf.nn.softmax, layer_name="prediction")

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Training started.")
        total_batch = int(len(mnist.train.labels) / batch_size)
        for epoch in range(epochs):
            avg_cost = 0
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
                _, c = sess.run([optimizer, cross_entropy], feed_dict={x: batch_x, y: batch_y})
                avg_cost += c / total_batch
            test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), " test accuracy: {:.3f}".format(test_acc))

        print("\nTraining complete!")
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))


if __name__ == "__main__":
    run_cnn()
