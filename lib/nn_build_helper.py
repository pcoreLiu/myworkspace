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