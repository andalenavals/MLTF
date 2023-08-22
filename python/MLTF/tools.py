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
logger = logging.getLogger(__name__)
import json
import os
import tensorflow as tf
import numpy as np


def check_history(hist, history_path, loss='loss',  reuse=True):
    data=[]
    if reuse:
        if os.path.isfile(history_path):
            logger.info("loading history %s"%(history_path))
            with open(history_path) as f:
                data = json.load(f)
            
    history_list= data+ hist.history[loss]
    with open(history_path, 'w') as outfile:
        json.dump(history_list, outfile)
    return history_list

def check_history_batch(hist, history_path, reuse=True):
    history_list= hist
    if reuse:
        if os.path.isfile(history_path):
            logger.info("loading history %s"%(history_path))
            with open(history_path) as f:
                data = json.load(f)
            history_list= np.append(data, hist, axis=1).tolist()            
            
    with open(history_path, 'w') as outfile:
        json.dump(history_list, outfile)
    return history_list


class BCP(tf.keras.callbacks.Callback):
    batch_loss = []
    def on_train_begin(self, logs={}):
        self.batch_loss = []
    def on_batch_end(self, batch, logs={}):
        self.batch_loss.append(logs.get('loss'))
        

