import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from lib import nn_build_helper


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

    conv_1 = nn_build_helper.create_conv_layer(x_shaped, 1, 32, [5, 5], name="conv_layer_1")  # output: 28*28*32
    conv_1 = nn_build_helper.create_max_pool(conv_1, [2, 2])  # output: 14*14*32
    conv_2 = nn_build_helper.create_conv_layer(conv_1, 32, 64, [5, 5], name="conv_layer_2")  # output: 14*14*64
    conv_2 = nn_build_helper.create_max_pool(conv_2, [2, 2])  # output: 7*7*64

    flattened = tf.reshape(conv_2, [-1, 7 * 7 * 64])

    dense_layer_1 = nn_build_helper.add_layer(flattened, flattened.shape[1].value, 1568,
                                              activation_function=tf.nn.relu, layer_name="dense_layer_1")
    dense_layer_2 = nn_build_helper.add_layer(dense_layer_1, 1568, 392,
                                              activation_function=tf.nn.relu, layer_name="dense_layer_2")
    prediction = nn_build_helper.add_layer(dense_layer_2, 392, 10,
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
