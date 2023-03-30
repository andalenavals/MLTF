
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['KMP_WARNINGS'] = 'off'

import logging

import MLTF


import tensorflow as tf
tf.enable_eager_execution()
tf.get_logger().setLevel(logging.ERROR)


import numpy as np

class Test_model:

    def test_model(self):

        model = MLTF.models.get_model()

        data = tf.zeros((4, 3, 2)) # 2 features
        out = model(data)

        #print(model.summary(line_length=100))
        # How many parameters: 2 features, 2 layers of 5, one output
        # Gives 51 params, including biases.
        assert model.count_params() == 51

        model = MLTF.models.get_model(hidden_sizes=(10, 10, 10))
        data = tf.zeros((4, 3, 3)) # 3 features
        out = model(data)
        assert model.count_params() == 271
