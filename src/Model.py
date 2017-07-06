import tensorflow as tf


def conv_layer(input, size_in, size_out, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def fc_layer(input, size_in, size_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        act = tf.nn.relu(tf.matmul(input, w) + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act


def model(x):
    with tf.name_scope('model'):
        x_image = tf.reshape(x, [-1, 64, 64, 3])
        tf.summary.image('input', x_image, 3)

        conv1 = conv_layer(x_image, 3, 32, "conv1")
        conv_out = conv_layer(conv1, 32, 64, "conv2")

        flattened = tf.reshape(conv_out, [-1, 16 * 16 * 64])

        fc1 = fc_layer(flattened, 16 * 16 * 64, 32, "fc1")
        logits = fc_layer(fc1, 32, 4, "fc2")

        return logits


