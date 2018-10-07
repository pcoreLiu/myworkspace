import tensorflow as tf
# Define a layer
def add_layer(inputs, in_size, out_size, activation_function=None, layer_name=""):
    # add one more layer and return the output of this layer
    # layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size], stddev=0.03), name='W')
            tf.summary.histogram(layer_name + '/weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.01, name='b')
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs


def compute_accuracy(test_data, test_data_label, sess, trained_model, model_data_place_holder):
    # prediction
    y_pre = sess.run(trained_model, feed_dict={model_data_place_holder: test_data})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(test_data_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print("Accuracy: ", sess.run(accuracy))
    return sess.run(accuracy)
