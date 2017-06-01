import tensorflow as tf

hello = tf.constant('Hello, world')

with tf.Session() as sess:
    print(sess.run(hello))
