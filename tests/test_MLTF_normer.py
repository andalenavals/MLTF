
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['KMP_WARNINGS'] = 'off'

import logging

import MLTF

import numpy as np

class Test_normer:

    @classmethod
    def setup_class(cls):

        cls.data = np.ma.array([
                [[10.0, 0.1], [11.0, 0.2], [9.0, 0.1]],
                [[40.0, 0.1], [41.0, 0.5], [42.0, 0.1]]
            ], mask = [
                [[False, False], [False, False], [False, False]],
                [[False, False], [False, True], [False, False]]
            ])

        assert cls.data.shape == (2,3,2)


    def test_normer(self):
        normer = MLTF.normer.Normer(self.data)
        normdata = normer(self.data)
        assert np.all(normdata<=1.0)
        reconstructed_data = normer.denorm(normdata)
        diff=reconstructed_data-self.data
        assert np.all(diff==0.0)

