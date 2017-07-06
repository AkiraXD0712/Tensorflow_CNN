import tensorflow as tf
import numpy as np
from Input import distorted_inputs
from Model import model

SAVE_PATH = 'C:\\Users\Akira.DESKTOP-HM7OVCC\Desktop\data\\model.ckpt'


def train():

    # Import data
    img_batch, label_batch = distorted_inputs('C:\\Users\Akira.DESKTOP-HM7OVCC\Desktop\data\\', 20)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 64, 64, 3], name='input')

    # Define loss and optimizer
    y = tf.placeholder(tf.float32, [None, 4], name='label')

    # Build the graph for the deep net
    logits = model(x)

    with tf.name_scope("xent"):
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=y), name="loss")
        tf.summary.scalar("loss", loss)

    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    # tf.reset_default_graph()
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('C:\\Users\Akira.DESKTOP-HM7OVCC\Desktop\data\\', sess.graph)
    tf.train.start_queue_runners(sess=sess)

    for i in range(500):
        img, label = sess.run([img_batch, label_batch])
        if i % 10 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: img, y: label})
            print('step %d, training accuracy %g' % (i, train_accuracy))
            result = sess.run(merged, feed_dict={x: img, y: label})
            writer.add_summary(result, i)
        sess.run(train_step, feed_dict={x: img, y: label})
    saver.save(sess, SAVE_PATH)
    print('Model stored...')


if __name__ == '__main__':
    train()

