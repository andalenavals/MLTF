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
        else:
            logger.info("%s is not a file"%(history_path))
            
    history_list= data+ hist.history[loss]
    with open(history_path, 'w') as outfile:
        json.dump(history_list, outfile)
    return history_list

def check_history_batch(hist, history_path, reuse=True):
    if reuse:
        if os.path.isfile(history_path):
            logger.info("loading history %s"%(history_path))
            with open(history_path) as f:
                data = json.load(f)
            history_list= np.append(data, hist, axis=1).tolist()
        else:
            logger.info("%s is not a file"%(history_path))
            history_list= hist
            
    with open(history_path, 'w') as outfile:
        json.dump(history_list, outfile)
    return history_list


class BCP(tf.keras.callbacks.Callback):
    batch_accuracy = [] # accuracy at given batch
    batch_loss = [] # loss at given batch    
    def __init__(self):
        super(BCP,self).__init__() 
    def on_train_batch_end(self, batch, logs=None):                
        BCP.batch_accuracy.append(logs.get('accuracy'))
        BCP.batch_loss.append(logs.get('loss'))
    def __len__(self):
        #len(self.x) is the length of your input features
        return math.ceil(len(self.x) / self.batch_size) 
