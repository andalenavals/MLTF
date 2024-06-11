"""
A module including custom activations functions
"""
import tensorflow as tf

import logging
logger = logging.getLogger(__name__)

@tf.function
def maxzerosigmoid(x):
    condition= tf.greater(x, 0.0)
    result = tf.cond(condition, lambda: tf.keras.backend.sigmoid(x), lambda: 0.0)
    return result

@tf.function
def maxzerotanh(x):
    if x>0.0: return tf.keras.backend.tanh(tf.constant(x, dtype=tf.float32))
    else: return 0.0

@tf.function
def binarystep(x):
    condition= tf.greater(x, 0.0)
    result = tf.where(condition, 1.0, 0.0)
    return result


list=["maxzerosigmoid","maxzerotanh","binarystep"]
