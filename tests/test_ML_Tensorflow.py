
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['KMP_WARNINGS'] = 'off'

import logging

import ML_Tensorflow


import tensorflow as tf
tf.enable_eager_execution()
tf.get_logger().setLevel(logging.ERROR)

def test_info():
    ML_Tensorflow.info.info()
    
    
