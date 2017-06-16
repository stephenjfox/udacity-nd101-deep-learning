import tensorflow as tf

# represent the expression 10 / 2 - 1 in TensorFlow
x = tf.constant(10)
y = tf.constant(2)
z = tf.truncatediv(x, y) - tf.constant(1)

with tf.Session() as sess:
    print(sess.run(z))
