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
File: ML_Tensorflow/examples/regression/inverse/noise_regression.py

Created on: 13/09/22
Author: Andres Navarro
"""

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(
        "ignore", category=DeprecationWarning,
        message=r"tostring\(\) is deprecated\. Use tobytes\(\) instead\.")


import sys, os
import MLTF
import numpy as np
import random
import pickle
import matplotlib
import astropy
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')
import data
from timeit import default_timer as timer

import tensorflow as tf

if float(tf.__version__[:3]) < 2.0:
    print("Using eager execution")
    #tf.compat.v1.enable_eager_execution()
    tf.enable_eager_execution()
tf.keras.mixed_precision.set_global_policy("mixed_float16")# predictions
PRECISION="float16"

# find if GPU or CPU is used
#tf.debugging.set_log_device_placement(True)
    
import logging
logger = logging.getLogger(__name__)



#model_kwargs={'loss_name':'msb', 'use_mask': True, 'hidden_sizes':(5,5), 'activation':'sigmoid', 'layer':MLTF.layer.TfbilacLayer}
model_kwargs={'loss_name':'msb', 'use_mask': True, 'hidden_sizes':(200,200,200,200), 'activation':'sigmoid', 'layer':tf.keras.layers.Dense, 'dtype':PRECISION}
#model_kwargs={'loss_name':'msb', 'use_mask': True, 'hidden_sizes':(5,5), 'activation':'sigmoid', 'layer':tf.keras.layers.Dense}
NFEATS=2

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Main script for training and validating fiducial simulations')
    parser.add_argument('--workdir',
                        default='out', 
                        help='diractory of work')
    parser.add_argument('--train', default=False,
                        action='store_const', const=True, help='Train point estimate')
    parser.add_argument('--validate', default=False,
                        action='store_const', const=True, help='Validate point estimate')
    parser.add_argument('--finetune', default=False,
                        action='store_const', const=True, help='Use SGD for getting as low as possible in the converged region')
    parser.add_argument('--batch_size', default=None,
                        help='Size of minibatches ')
   
    args = parser.parse_args()

    return args


def f(x): return np.sqrt(1.0 + x**2)
def g(x): return x**3

def train(features, targets, trainpath, checkpoint_path=None, reuse=True, finetune=True, epochs=1000, validation_split=None, validation_data=None, batch_size=None):
    mask =np.all(~features.mask,axis=2,keepdims=True)
    caseweights=None
    
    #features_normer=MLTF.normer.Normer(features, type=inputtype)
    #features=features_normer(features)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                     monitor='loss',#"loss"
                                                     save_weights_only=True,
                                                     save_best_only= True,
                                                     verbose=1, 
                                                     save_freq='epoch')

    batch_callback=MLTF.tools.BCP()
    redlr_callback=tf.keras.callbacks.ReduceLROnPlateau( monitor="loss",
                                                         factor=0.1,
                                                         patience=1000,
                                                         verbose=1,
                                                         mode="auto",
                                                         min_delta=1e-10,
                                                         cooldown=0,
                                                         min_lr=0,)

    #input_shape=features[0].shape #(nreas, nfeas)
    input_shape=(None, features[0].shape[1])

    training_data=[features.data, targets, mask]
    
    
    USEGPU=True
    if USEGPU:
        #gpus=["GPU:0", "GPU:1"]
        gpus=["GPU:1"]
        #gpus = tf.config.list_physical_devices('GPU')
        strategy = tf.distribute.MirroredStrategy(gpus)
        with strategy.scope():
            opt=tf.keras.optimizers.Adam(learning_rate=0.01)
            #opt=tf.keras.optimizers.SGD(learning_rate=0.1)
            model=MLTF.models.create_model(input_shape, **model_kwargs)
            model.compile(loss=None, optimizer=opt, metrics = [])
    else:
        opt=tf.keras.optimizers.Adam(learning_rate=0.01)
        #opt=tf.keras.optimizers.SGD(learning_rate=0.1)
        model=MLTF.models.create_model(input_shape, **model_kwargs)
        model.compile(loss=None, optimizer=opt, metrics = [])
        if os.path.isfile(checkpoint_path+'.index') & reuse:
            logger.info("loading checkpoint weights")
            model.load_weights(checkpoint_path)
            model.compile(loss=None, optimizer=opt, metrics = [])

    start = timer()
    hist = model.fit(training_data, None, epochs=epochs, verbose=2,
                     shuffle=True, batch_size=batch_size,
                     validation_split=validation_split,
                     validation_data=validation_data,
                     callbacks=[cp_callback,batch_callback,redlr_callback])
    duration = timer() -start
    print("This operation took %.3f ms" % (1e3*duration))
    
    

    

    history_path=os.path.join(trainpath, "history.txt")
    history = MLTF.tools.check_history(hist, history_path, loss='loss',reuse=reuse)
    if (validation_data is not None)|(validation_split is not None):
        history_path=os.path.join(trainpath, "history_val.txt")
        history_val = MLTF.tools.check_history(hist, history_path, loss='val_loss',reuse=reuse)
    if batch_size is not None:
        history_path=os.path.join(trainpath, "history_batches.txt")
        batch_hist=np.array(np.split(np.array(batch_callback.batch_loss), epochs)).T.tolist()
        history_batch= MLTF.tools.check_history_batch(batch_hist, history_path, reuse=reuse)
  
    if finetune:
        reuse=True
        fine_batch_callback=MLTF.tools.BCP()
        model=MLTF.models.create_model(input_shape, **model_kwargs)
        opt=tf.keras.optimizers.SGD(learning_rate=0.1)
        model.compile(loss=None, optimizer=opt, metrics = [])
        if os.path.isfile(checkpoint_path+'.index') & reuse:
            logger.info("loading checkpoint weights")
            model.load_weights(checkpoint_path)
        hist = model.fit(training_data, None, epochs=epochs, verbose=2, 
                         shuffle=True, batch_size=batch_size,
                         validation_split=validation_split, validation_data=validation_data,
                         callbacks=[cp_callback, fine_batch_callback])

        history_path=os.path.join(trainpath, "history.txt")
        history = MLTF.tools.check_history(hist, history_path, loss='loss',reuse=reuse)
        
        if (validation_data is not None)|(validation_split is not None):
            history_path=os.path.join(trainpath, "history_val.txt")
            history_val = MLTF.tools.check_history(hist, history_path, loss='val_loss',reuse=reuse)
        if batch_size is not None:
            history_path=os.path.join(trainpath, "history_batches.txt")
            batch_hist=np.array(np.split(np.array(fine_batch_callback.batch_loss), epochs)).T.tolist()
            history_batch= MLTF.tools.check_history_batch(batch_hist, history_path, reuse=reuse)

    xscalelog=True
    yscalelog=True
    filename=os.path.join(trainpath, "history_train_and_val.png")
    fig, ax = plt.subplots()
    MLTF.plot.plot_history_ax(ax,history, xscalelog=xscalelog, yscalelog=yscalelog, label="Training set")
    if (validation_data is not None)|(validation_split is not None): MLTF.plot.plot_history_ax(ax,history_val, xscalelog=xscalelog, yscalelog=yscalelog, label="Validation set")    
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=20)
    plt.ylim(0.5*min(history), 1.5*max(history))
    plt.tight_layout()
    plt.savefig(filename,dpi=200)
    plt.close()

    filename=os.path.join(trainpath, "history_train_and_batches.png")
    fig, ax = plt.subplots()
    MLTF.plot.plot_history_ax(ax,history, xscalelog=xscalelog, yscalelog=yscalelog, label="Training set")
    if batch_size is not None:
        for i, h in enumerate(history_batch):
            MLTF.plot.plot_history_ax(ax,h, xscalelog=xscalelog, yscalelog=yscalelog, label="Minibatch %i"%(i+1))    
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=20)
    plt.ylim(0.5*min(history), 1.5*max(history))
    plt.tight_layout()
    plt.savefig(filename,dpi=200)
    plt.close()

    
    logger.info("***--- TRAINING FINISHED --- ***")

    

def validate(features, targets, checkpoint_path, valpath, targets_normer):
    
    MAX_NPOINTS=5000
    
    targets=targets_normer.denorm(targets)
    mask =np.all(~features.mask,axis=2,keepdims=True)

    input_shape=features[0].shape #(nreas, nfeas)
    model=MLTF.models.create_model(input_shape, **model_kwargs )
    model.load_weights(checkpoint_path)
    if float(tf.__version__[:3]) >2.0:
        preds = model([features, tf.constant(0, shape=(features.shape[0], 1 , 1)), tf.constant(0, shape=features.shape)]).numpy()
    elif float(tf.__version__[:3]) <2.0:
        preds = model(features.astype(PRECISION)).numpy()
    preds=np.ma.array(preds,mask=~mask)
    preds=targets_normer.denorm(preds)

    val_biases_msb = np.mean(preds, axis=1, keepdims=True) - targets

    loss_val_direct=np.mean(np.square(val_biases_msb))    
    mask=tf.convert_to_tensor((mask*1.0).astype(PRECISION))
    loss_val=MLTF.loss_functions.msb(tf.convert_to_tensor(targets.astype(PRECISION)), tf.convert_to_tensor(preds.astype(PRECISION)), mask=mask)
    print(loss_val_direct,loss_val.numpy())
    
    filename=os.path.join(valpath, "bias_vs_targets.png")
    MLTF.plot.color_plot(np.ma.array(targets[:,0,0],mask=False) ,val_biases_msb[:,0,0], None,False, r"$\theta$" ,r"$\langle \hat{\theta} - \theta \rangle$", "" , title="", ftsize=18,cmap="gnuplot", filename=filename, npoints_plot=MAX_NPOINTS, linreg=True)
    

        
    logger.info("***--- VALIDATING FINISHED --- ***")


def test(features, targets, checkpoint_path, func, path, features_test, targets_normer, features_normer):
    #features: training features
    #targets: training targets
    
    mask_test =np.all(~features_test.mask,axis=2,keepdims=True)       
    
    #Loading model
    input_shape=features_test[0].shape #(nreas, nfeas)
    model=MLTF.models.create_model(input_shape, **model_kwargs)
    model.load_weights(checkpoint_path)
    if float(tf.__version__[:3]) >2.0:
        test_preds = model.predict([features_test, tf.constant(0, shape=features_test.shape), tf.constant(0, shape=features_test.shape)])
    elif float(tf.__version__[:3]) <=2.0:
        test_preds = model(features_test.astype(PRECISION)).numpy()
    test_preds=np.ma.array(test_preds,mask=~mask_test)
    test_preds=targets_normer.denorm(test_preds)


    #Selecting few training data points for the plot
    npoints=10000
    targets_1d = np.ma.concatenate(targets_normer.denorm(targets).T)[0]
    features_1d = np.ma.concatenate(features_normer.denorm(features)[:,:,0].T)
    showtrainindices = np.arange(targets_1d.size)
    np.random.shuffle(showtrainindices)
    showtrainindices = showtrainindices[:npoints]
    targets_1d= targets_1d[showtrainindices]
    features_1d = features_1d[showtrainindices]

    features_test=features_normer.denorm(features_test)

    # True function    
    trutheta = np.linspace( -1.0, 2.2, 100)
    trud = func(trutheta)
    color_cycle = ["#1b9e77", "#d95f02", "#7570b3"]

    plt.plot(targets_1d, features_1d, marker=".", color="gray", ls="None", ms=2, label="Training data samples")
    plt.plot(test_preds[:,:,0], features_test[:,:,0], ls="-", color=color_cycle[2], label="Trained with %s"%(model_kwargs['loss_name']), lw=1.5, alpha=0.5)
    plt.plot(trutheta, trud, ls="-", color="black", dashes=(5, 5), lw=2.0, label=r"$d = \sqrt{1 + \theta^2}$")
    plt.xlabel(r"$\theta$ $\mathrm{and}$ $\hat{\theta}$", fontsize=18)
    plt.ylabel(r"$d$", fontsize=18)
    plt.legend(loc=2, fontsize=12, markerscale=4, numpoints=1)
    plt.xlim(-1.2, 2.4)
    plt.ylim(0.5, 3.0)
    plt.tight_layout()
    filename=os.path.join(path, "test.png")
    plt.savefig(filename, dpi=200)

    logger.info("***--- TESTING FINISHED --- ***")
    
    
def make_dir(dirname):
    try:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    except OSError:
        if not os.path.exists(dirname): raise
        
def main():    
    args = parse_args()
    loggerformat='PID %(process)06d | %(asctime)s | %(levelname)s: %(name)s(%(funcName)s): %(message)s'
    #logging.basicConfig(format=loggerformat, level=logging.DEBUG)
    logging.basicConfig(format=loggerformat, level=logging.INFO)
    #logging.basicConfig(format=loggerformat, level=logging.NOTSET)
    
    outpath = os.path.expanduser(args.workdir)
    make_dir(outpath)
    example_path='point_noise_regression_%ifeats_%s'%(NFEATS, model_kwargs['loss_name'])
    
    trainingpath = os.path.expanduser(os.path.join(outpath,example_path, "training"))
    make_dir(trainingpath)
    validationpath = os.path.expanduser(os.path.join(outpath,example_path,"validation"))
    make_dir(validationpath)
    
    normerspath = os.path.expanduser(os.path.join(outpath, "data","normers"))
    make_dir(normerspath)
    trainingcat=os.path.join(outpath, "data", "traincat%s_%i.pkl"%(sys.version[:5], NFEATS))
    trainingvalcat=os.path.join(outpath, "data", "trainvalcat%s_%i.pkl"%(sys.version[:5], NFEATS))
    validationcat=os.path.join(outpath, "data", "valcat%s_%i.pkl"%(sys.version[:5], NFEATS))
    testcat=os.path.join(outpath, "data", "testcat%s_%i.pkl"%(sys.version[:5], NFEATS))

    checkpoint_path=os.path.join(trainingpath, "simple_regression.ckpt")

    ncases=50
    nreas=100
    nmsk_obj=50
    features,targets=data.makedata(ncases, nreas, f, nmsk_obj, filename=trainingcat, nfeats=NFEATS)
    features_normer=MLTF.normer.Normer(features, type="01") #sa1
    features=features_normer(features)
    targets_normer=MLTF.normer.Normer(targets, type="01")
    targets=targets_normer(targets)
    
    features_val,targets_val=data.makedata(ncases, nreas+100, f, nmsk_obj, filename=trainingvalcat, nfeats=NFEATS)
    features_val=features_normer(features_val)
    targets_val=targets_normer(targets_val)
    validation_data= ([features_val.data ,targets_val, np.all(~features_val.mask,axis=2,keepdims=True)],None)
    #validation_data= None
    #validation_split=0.3
    validation_split=None
    logger.info("Data was done")
    
    train(features,targets, trainingpath, checkpoint_path, reuse=True ,epochs=100, validation_data=validation_data, validation_split=validation_split, finetune=args.finetune, batch_size=args.batch_size )

    features_test=data.maketestdata(ncases=100, nfeats=NFEATS)
    features_test=features_normer(features_test)
    features_val,targets_val=data.makedata(ncases, nreas, f, nmsk_obj, filename=validationcat)
    features_val=features_normer(features_val)
    targets_val=targets_normer(targets_val)
    validate(features_val, targets_val, checkpoint_path, validationpath, targets_normer )
    test(features, targets,checkpoint_path, f, validationpath, features_test, targets_normer, features_normer)



    
      
if __name__ == "__main__":
    main()
