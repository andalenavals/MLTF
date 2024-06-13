#
# Copyright (C) 2012-2020 Euclid Science Ground Segment
#
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

"""
File: python/SHE_KerasML/utils.py

Created on: 13/12/19
Author: Malte Tewes
"""

import logging
import numpy as np


logger = logging.getLogger(__name__)



class Normer:
    """
    A Normer provides methods to "normalize" and later "denormalize" a numpy array:
    - linearly rescaling it to be within 0 and 1 (type="01"),
    - to be within -1 and 1 (type="-11"),
    - around 0 with a std of 1 (type="std"),
    - or just factor-scaling it so that the maximum absolute value is 1 (type="sa1").

    For each feature, the same normalization is applied to all realizations of all cases.
    The details of the normalization are kept in the Normer object, so that one can simply use the same
    object to denorm stuff afterwards.

    This works with 3D and 2D arrays, with indices (case, rea, feat)
    or (case, feat).

    All methods work with masked arrays: masked values are ignored
    when computing the normer parameters, and the mask itself is unchanged by
    norming or denorming.
    """

    def __init__(self, x=None, a=None, b=None, type="-11"):
        """
        Does *not* normalize anything, just determines the normalization parameters!
        """

        self.type = type
        
        if x is not None:
            if isinstance(x, np.ma.MaskedArray):
                logger.debug("Building Normer of type '{}' with a masked array of shape {} and {} masked values".format(
                    self.type, str(x.shape), np.sum(x.mask)))
                # np.ma.set_fill_value(x, 1.e20) # To notice things if the array gets filled by error.
            elif isinstance(x, np.ndarray):
                logger.debug("Building Normer of type '{}' with an unmasked array of shape {}".format(self.type, x.shape))
            else:
                raise ValueError("x is not a numpy array")

            if x.ndim not in (2, 3):
                raise ValueError("Cannot handle this array shape")

            if type in ["01", "-11"]:
                # For this, we need to compute the min and max value for every feature, along the realizations and cases.

                if x.ndim == 3:  # If we have several realizations:
                    mins = np.min(np.min(x, axis=1), axis=0)  # Computes the min along reas and cases.
                    dists = np.max(np.max(x, axis=1), axis=0) - mins


                # All these np.min, np.max, np.mean, np.std work as expected also with masked arrays.
                elif x.ndim == 2:
                    mins = np.min(x, axis=0)  # Only along cases
                    dists = np.max(x, axis=0) - mins

                assert mins.shape == (x.shape[-1],) # Only features are left
                assert dists.shape == (x.shape[-1],)

                self.a = mins
                self.b = dists

            elif type == "sa1":
                # We only rescale the values so that the max amplitude is 1. This ensures that signs are kept.

                if x.ndim == 3:
                    scales = np.max(np.max(np.fabs(x), axis=1), axis=0)
                elif x.ndim == 2:
                    scales = np.max(np.fabs(x), axis=0)

                assert scales.ndim == 1  # Only the "feature" dimension remains.
                # assert scales.shape == (x.shape[-1])

                self.b = scales
                self.a = np.zeros(scales.shape)

            else:
                raise RuntimeError("Unknown Normer type")

            # logger.debug(str(self))
        else:
            self.a=a
            self.b=b
            

    def __str__(self):
        return "Normer of type '{self.type}': a={self.a}, b={self.b}".format(self=self)



    def __call__(self, x):
        """
        Returns the normalized data.
        """

        logger.debug("Normalizing array of shape {} with normer-type '{}'".format(x.shape, self.type))
        if x.ndim not in (2, 3):
            raise ValueError("Cannot handle this array shape")

        atiled = tileToShape(self.a, x)
        btiled = tileToShape(self.b, x)

        res = (x - atiled) / btiled

        if self.type == "-11":
            res = 2.0 * res - 1.0

        return res

    def denorm(self, x):
        """
        Denorms the data
        """

        if x.ndim not in (2, 3):
            raise ValueError("Cannot handle this array shape")


        if self.type == "-11":
            res = (x + 1.0) / 2.0
        else:
            res = x + 0.0

        atiled = tileToShape(self.a, x)
        btiled = tileToShape(self.b, x)

        res = res * btiled + atiled

        return res


def tileToShape(ab, x):
    """
    - ab: an array of shape (feat), typically holding the "a" and "b" terms of a normer.
    - x: an array (case, rea, feat) or (case, feat) to which the ab-array should be inflated.
    """
    assert ab.ndim == 1
    nfeat = ab.shape[0]
    assert nfeat == x.shape[-1]
    ncase = x.shape[0]

    if x.ndim == 2:
        abtiled = np.tile(ab, (ncase, 1))

    elif x.ndim == 3:
        nrea = x.shape[1]
        abtiled = np.tile(ab, (ncase, nrea, 1))

    return abtiled
