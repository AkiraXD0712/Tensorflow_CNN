import tensorflow as tf
from Train import SAVE_PATH
from Model import model

x = tf.placeholder(tf.float32, [None, 64, 64, 3])

# saver = tf.train.Saver()
sess = tf.Session()
# First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('C:\\Users\Akira.DESKTOP-HM7OVCC\Desktop\data\model.ckpt.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))

# Access saved Variables directly
print(sess.run('conv1/B'))


