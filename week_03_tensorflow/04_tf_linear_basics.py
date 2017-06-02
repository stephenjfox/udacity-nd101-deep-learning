# Solution is available in the other "quiz_solution.py" tab
import tensorflow as tf

## Just some thoughts
## If we're working with a reduced MNIST (of 0s, 1s, and 2s), there would only be 3 labels
## Because we would then review the squashed 28x28 -> 784, n_features = 784
def get_weights(n_features, n_labels):
    """
    Return TensorFlow weights
    :param n_features: Number of features
    :param n_labels: Number of labels
    :return: TensorFlow weights
    """
    return tf.Variable(tf.truncated_normal((n_features, n_labels)))


def get_biases(n_labels):
    """
    Return TensorFlow bias
    :param n_labels: Number of labels
    :return: TensorFlow bias
    """
    return tf.Variable(tf.zeros(n_labels))


def linear(input, w, b):
    """
    Return linear function in TensorFlow
    :param input: TensorFlow input
    :param w: TensorFlow weights
    :param b: TensorFlow biases
    :return: TensorFlow linear function
    """
    return tf.matmul(input, w) + b
