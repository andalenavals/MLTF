
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

## Unmasked data
class Test_layer:

    @classmethod
    def setup_class(cls):

        # Create dummy data, in the format (case, rea, feature)
        data = np.array([
            [[0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0]],
        ])

        data[1, 1, 1] = 5.0
        assert data.shape == (4, 3, 2)

        data = tf.convert_to_tensor(data, dtype=tf.float32)
        # data = tf.zeros((4, 3, 2))
        #print(data)
        cls.data = data


    def test_shape(self):
        #data = get_dummy_data()
        # Now create a layer
        lay = MLTF.layer.TfbilacLayer(10)

        assert lay.compute_output_shape(self.data.shape) == (4, 3, 10)

        out_data = lay(self.data)
        assert out_data.shape ==(4, 3, 10)
        assert len(lay.variables) == 2 # Is empty before build(), so this works only after above call

        #print(out_data)
