import numpy as np
import tensorflow as tf

num_class = 4

array = np.array([3])
array = tf.one_hot(array, depth=num_class, axis=-1)
print(array)
sess = tf.Session()
print(sess.run(array))

