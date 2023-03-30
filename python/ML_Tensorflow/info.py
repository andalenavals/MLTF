
import logging

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

def info():
    print("TensorFlow version: {}".format(tf.__version__))
    print("tf.keras version: {}".format(tf.keras.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))
