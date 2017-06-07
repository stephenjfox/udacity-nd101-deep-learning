import tensorflow as tf

# data to back the application
softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

# placeholders - this declares the unit data type (aka the type of a tensor at the lowest dimension)
softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)

# This is why I love TensorFlow: what is crossentropy error?

# The negated summation of the labels, multiplied by the natural log of the prediction

# Natural log of the prediction
ln_prediction = tf.log(softmax)

multiplied = one_hot * ln_prediction

# summation
summation = -tf.reduce_sum(multiplied)

with tf.Session() as sess:
    output = sess.run(summation, feed_dict={
                                            softmax: softmax_data,
                                            one_hot: one_hot_data })
    print("The (crossentropy) error =", output)
