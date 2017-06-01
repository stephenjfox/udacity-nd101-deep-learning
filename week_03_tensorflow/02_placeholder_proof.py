import tensorflow as tf

x = tf.placeholder(tf.float32)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={ x: 123.2 })
    print(output) # haha at floating point inaccuracy
