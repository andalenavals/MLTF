import warnings
warnings.filterwarnings("ignore")
import os
os.environ['KMP_WARNINGS'] = 'off'

import logging

import MLTF

import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
tf.get_logger().setLevel(logging.ERROR)


import numpy as np

## Unmasked data
class Test_loss:

    @classmethod
    def setup_class(cls):

        cls.truths = tf.convert_to_tensor([
            [1], [2], [3], [4]
            ], dtype=tf.float32)

        cls.three_d_truths = tf.convert_to_tensor([
            [[1]], [[2]], [[3]], [[4]]
            ], dtype=tf.float32)

        cls.preds = tf.convert_to_tensor([
            [[0.5], [1], [1.5]], [[1.5], [2], [2.5]], [[2.9], [3.0], [3.1]], [[3.8], [4.0], [4.2]]
            ], dtype=tf.float32)

        cls.wpreds = tf.convert_to_tensor([
            [[1],[2],[3]], [[1],[1],[1]], [[1],[1],[1]], [[1],[1],[1]]
            ], dtype=tf.float32)

        cls.mpreds = tf.convert_to_tensor([
            [[-1],[0],[1./3]], [[0],[0],[0]], [[0],[0],[0]], [[0],[0],[0]]
            ], dtype=tf.float32)

        assert cls.truths.shape == (4, 1)
        assert cls.three_d_truths.shape == (4, 1, 1)
        assert cls.preds.shape == (4, 3, 1)
        assert cls.wpreds.shape == (4, 3, 1)
        assert cls.mpreds.shape == (4, 3, 1)


    def test_msb(self):
        msb = MLTF.loss_functions.msb
        val = tf.keras.backend.get_value(msb(self.three_d_truths, self.preds))
        assert np.isclose(val, 0.0)
    def test_mse(self):
        mse = MLTF.loss_functions.mse
        val = tf.keras.backend.get_value(mse(self.three_d_truths, self.preds))
        assert np.isclose(val, 11./120)
    def test_mswb(self):
        mswb = MLTF.loss_functions.mswb
        val = tf.keras.backend.get_value(mswb(self.three_d_truths, self.wpreds, self.preds))
        assert np.isclose(val, 0.006944444444444451)
    def test_msmb(self):
        msmb = MLTF.loss_functions.msmb
        val = tf.keras.backend.get_value(msmb(self.three_d_truths, self.mpreds, self.preds))
        assert np.isclose(val, 0.0)
    def test_mswcb(self):
        mswcb = MLTF.loss_functions.mswcb
        val = tf.keras.backend.get_value(mswcb(self.three_d_truths, self.wpreds, self.mpreds, self.preds))
        assert np.isclose(val, 1./36)
