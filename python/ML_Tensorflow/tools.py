import logging
logger = logging.getLogger(__name__)
import json
import os
import tensorflow as tf

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

class BCP(tf.keras.callbacks.Callback):
    batch_accuracy = [] # accuracy at given batch
    batch_loss = [] # loss at given batch    
    def __init__(self):
        super(BCP,self).__init__() 
    def on_train_batch_end(self, batch, logs=None):                
        BCP.batch_accuracy.append(logs.get('accuracy'))
        BCP.batch_loss.append(logs.get('loss'))
