# This library is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3.0 of the License, or (at your option)
# any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
#

import logging

import tensorflow as tf

logger = logging.getLogger(__name__)

class TfbilacLayer(tf.keras.layers.Layer):
    """ A custom dense layer handling cases and realizations and masks

    Inspiration:

    https://keras.io/layers/core/
    (for Maskign core layer!)

    https://keras.io/layers/writing-your-own-keras-layers/
    (see for multiple inputs!)

    """

    def __init__(self, nout, **kwargs):
        """nout is the number of nodes"""
        super(TfbilacLayer, self).__init__(**kwargs)
        self.nout = nout

    def build(self, input_shape):
        """input_shape is (ncase, nrea, nfeat)"""

        assert len(input_shape) == 3
        (ncase, nrea, nfeat) = input_shape
        
        if float(tf.__version__[:3]) >2.0:
            self.kernel = self.add_weight(
                name="kernel",
                shape=(nfeat, self.nout),
                initializer="uniform",
                trainable=True
                )
        elif float(tf.__version__[:3]) <2.0:
            self.kernel = self.add_variable(
                name="kernel",
                shape=(nfeat, self.nout),
                initializer="uniform",
                trainable=True
                )
        

        self.bias = self.add_weight(
                name="bias",
                shape=(self.nout,),
                initializer="zeros",
                trainable=True)

        # Finally, call this to set self.built = True
        # according to https://keras.io/layers/writing-your-own-keras-layers/
        super(TfbilacLayer, self).build(input_shape)


    def compute_output_shape(self, input_shape):
        """Returns (ncase, nrea, nout)"""
        assert len(input_shape) == 3
        (ncase, nrea, nfeat) = input_shape
        return (int(ncase), int(nrea), int(self.nout))

    def call(self, input):
        """Input is (ncase, nrea, nfeat), kernel is (nfeat, nout)"""
        return tf.matmul(input, self.kernel) + self.bias

    def get_config(self):
        config = super(TfbilacLayer, self).get_config()
        config.update({"units": self.nout})
        return config
