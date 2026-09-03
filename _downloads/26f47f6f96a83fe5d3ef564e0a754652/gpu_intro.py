#!/usr/bin/env python3

import tensorflow as tf

# Test if there are any GPUs available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Have Tensorflow output where computations are run
tf.debugging.set_log_device_placement(True)

# Create some tensors
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

# Print result
print(c)
