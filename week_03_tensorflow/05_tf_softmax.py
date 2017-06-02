import tensorflow as tf

logit_data = [2., 1., 0.1]
logits = tf.placeholder(tf.float32)

# setup a process to run on the placeholder
# When this is run, the C-code will inline the operation and produce output
softmax = tf.nn.softmax(logits)

with tf.Session() as sess:
    output = sess.run(softmax, feed_dict={logits: logit_data})
    print("We made it work!", output)
