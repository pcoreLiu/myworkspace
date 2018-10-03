import tensorflow as tf
import numpy as np
import matplotlib as plt

matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2], [2]])

product = tf.matmul(matrix1, matrix2)  # matrix multiply np.dot(m1, m2)


with tf.Session() as sess:
    print(sess.run(product))

x_data = np.random.rand(2000).astype(np.float32)
y_data = tf.square(x_data) * 0.7 + 0.3

Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

# loss = tf.losses.mean_squared_error(y_data, y)
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for step in range(10001):
        sess.run(train)
        if step % 10 == 0:
            print(step, sess.run(Weights), sess.run(biases))

















