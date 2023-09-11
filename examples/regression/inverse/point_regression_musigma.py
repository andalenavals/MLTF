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
File: MLTF/examples/regression/inverse/noise_regression.py

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

import tensorflow as tf
if float(tf.__version__[:3]) < 2.0:
    print("Using eager execution")
    #tf.compat.v1.enable_eager_execution()
    tf.enable_eager_execution()
    
import logging
logger = logging.getLogger(__name__)


model_kwargs={'loss_name':'nll_normal', 'use_mask': True, 'hidden_sizes':(3,), 'activation':'sigmoid', 'layer':tf.keras.layers.Dense}
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

def noise(n): return np.random.randn(n)
def f(x): return np.sqrt(1.0 + x**2)
def g(x): return x**3

def makedata(ncases, nreas, func, nmsk_obj=0, noise_scale=0.1, theta_min=0.25 , theta_max=2.0, filename=None, shuffle=True):

    if filename is not None:
        if os.path.exists(filename):
            logger.info("Catalog was already done")
            with open(filename, 'rb') as handle:
                cat= pickle.load(handle)
            features, targets=cat
            return features, targets       
        
    #targets
    thetas = np.linspace( theta_min, theta_max, ncases)
    targets=thetas.reshape(ncases,1, 1)

    # Features
    aux=[]
    for rea in range(nreas):
        nos=noise_scale*noise(ncases)
        if NFEATS==2:
            aux.append([func(thetas)+nos, nos])
        if NFEATS==1:
            aux.append([func(thetas)+nos])
    features=np.array(aux)
    features= np.transpose(features,axes=(2,0,1))

    # Mask
    nfeats=NFEATS
    if nmsk_obj==0:
        mask=np.full((ncases, nreas, nfeats), True)
        features=np.ma.array(features, mask=~mask)

    else:
        mask0=np.full((ncases, nreas, nfeats), True)
        assert (nmsk_obj <= ncases*nreas*nfeats), "Masking larger than data set"
        idxs=[]
        while len(idxs)<nmsk_obj:
            ele =[random.randint(0, ncases-1) ,random.randint(0, nreas-1), random.randint(0, nfeats-1)]
            if ele not in idxs:  idxs.append(ele)

        cases=np.array(idxs).T[0]
        reas=np.array(idxs).T[1]
        feas=np.array(idxs).T[2]
        mask0[cases,reas,feas]=False
        #making live difficult to the training
        features[cases,reas,feas]=-999
        
        mask=np.all(mask0,axis=2,keepdims=True)

        #If all realizations are null remove whole case
        bl_cases=np.where(np.sum(mask,axis=1, keepdims=True)==0)[0]
        features=np.delete(features, bl_cases,  axis=0 )
        mask0=np.delete(mask0, bl_cases,  axis=0 )
        targets=np.delete(targets, bl_cases,  axis=0 )
        
        features=np.ma.array(features, mask=~mask0)
        logger.info('Number of blacklisted cases: %i'%len(bl_cases))

    if shuffle:
        ncases=targets.shape[0]
        ind=np.random.choice(range(ncases),size=ncases, replace=False)
        targets=targets[ind]
        features=features[ind]
    logger.info("Data was done")
    with open(filename, 'wb') as handle:
            pickle.dump([features, targets], handle, -1)
    
    return features, targets

def maketestdata(ncases=100):
    func_val=np.linspace( 0.5, 3,ncases)
    if NFEATS==2:
        features_test = np.transpose(np.array([[ func_val, np.array([0]*ncases) ] for rea in range(1)]),axes=(2,0,1))
    elif NFEATS==1:
        features_test = np.transpose(np.array([[ func_val ] for rea in range(1)]),axes=(2,0,1))
    return features_test


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
                                                         patience=5000,
                                                         verbose=1,
                                                         mode="auto",
                                                         min_delta=1e-10,
                                                         cooldown=0,
                                                         min_lr=0,)

    #input_shape=features[0].shape #(nreas, nfeas)
    input_shape=(None, features[0].shape[1])

    
    opt=tf.keras.optimizers.Adam(learning_rate=0.01)
    #opt=tf.keras.optimizers.SGD(learning_rate=0.1)
    model=MLTF.models.create_musigma_model(input_shape, **model_kwargs)
    model.compile(loss=None, optimizer=opt, metrics = [])
    
    if os.path.isfile(checkpoint_path+'.index') & reuse:
        logger.info("loading checkpoint weights")
        model.load_weights(checkpoint_path)
        model.compile(loss=None, optimizer=opt, metrics = [])
    
    training_data=[features.data, targets, mask]
    
    hist = model.fit(training_data, None, epochs=epochs, verbose=2, 
                     shuffle=True, batch_size=batch_size,
                     validation_split=validation_split, validation_data=validation_data,
                     callbacks=[cp_callback, batch_callback,redlr_callback])

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
    yscalelog=False
    filename=os.path.join(trainpath, "history_train_and_val.png")
    fig, ax = plt.subplots()
    MLTF.plot.plot_history_ax(ax,history, xscalelog=xscalelog, yscalelog=yscalelog, label="Training set")
    if (validation_data is not None)|(validation_split is not None): MLTF.plot.plot_history_ax(ax,history_val, xscalelog=xscalelog, yscalelog=yscalelog, label="Validation set")    
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=20)
    #plt.ylim(0.5*min(history), 1.5*max(history))
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
    #plt.ylim(0.5*min(history), 1.5*max(history))
    plt.tight_layout()
    plt.savefig(filename,dpi=200)
    plt.close()

    
    logger.info("***--- TRAINING FINISHED --- ***")

    

def validate(features, targets, checkpoint_path, valpath, targets_normer):
    
    MAX_NPOINTS=5000
    
    targets=targets_normer.denorm(targets)
    mask =np.all(~features.mask,axis=2,keepdims=True)

    input_shape=features[0].shape #(nreas, nfeas)
    model=MLTF.models.create_musigma_model(input_shape, **model_kwargs )
    model.load_weights(checkpoint_path)
    if float(tf.__version__[:3]) >2.0:
        pred_mu, pred_var = model([features, tf.constant(0, shape=(features.shape[0], 1 , 1)), tf.constant(0, shape=features.shape)])#.numpy()
    elif float(tf.__version__[:3]) <2.0:
        pred_mu, pred_var = model(features.astype('float32')).numpy()
    pred_mu=np.ma.array(pred_mu,mask=~mask)
    pred_var=np.ma.array(pred_var,mask=~mask)
    pred_mu=targets_normer.denorm(pred_mu)

    val_biases_msb = np.mean(pred_mu, axis=1, keepdims=True) - targets

    loss_val_direct=np.mean(np.square(val_biases_msb))    
    mask=tf.convert_to_tensor((mask*1.0).astype('float32'))
    loss_val=MLTF.loss_functions.nll_normal(tf.convert_to_tensor(targets.astype('float32')), tf.convert_to_tensor(pred_mu.astype('float32')), tf.convert_to_tensor(pred_var.astype('float32')), mask=mask)
    print(loss_val_direct,loss_val.numpy())

    # since the normer is linear Var(f(X))=[f'(EX)]^2 Var(X)
    # n(x)=(x-a)/b; if type="-11" n(x)=2*(x-a)/b -1; where a is min(x), b= max(x)-a
    # we are usin denorm, therefore f(x) is the inverse of n(x)
    # n_inv(x)= bx+a; if type="-11"" n_inv(x)=0.5*b*(x+1)+a
    #preds_variance=np.ma.array(preds_variance,mask=~mask)


    fprima=1./targets_normer.b
    if targets_normer.type=="-11": fprima*=2
    preds_variance=pred_var/(fprima**2)
    nreas=np.sum(~preds_variance.mask,axis=1) 
    val_biases_msb_err = (1/nreas)*np.ma.sqrt(np.ma.sum(preds_variance, axis=1))

    
    filename=os.path.join(valpath, "bias_vs_targets.png")
    MLTF.plot.color_plot(np.ma.array(targets[:,0,0],mask=False) ,val_biases_msb[:,0,0],None,False, r"$\theta$" ,r"$\langle \hat{\theta} - \theta \rangle$", "" , yerr=val_biases_msb_err[:,0], title="", ftsize=18,cmap="gnuplot", filename=filename, npoints_plot=MAX_NPOINTS, linreg=True)
    

        
    logger.info("***--- VALIDATING FINISHED --- ***")


def test(features, targets, checkpoint_path, func, path, features_test, targets_normer, features_normer):
    #features: training features
    #targets: training targets
    
    mask_test =np.all(~features_test.mask,axis=2,keepdims=True)       
    
    #Loading model
    input_shape=features_test[0].shape #(nreas, nfeas)
    model=MLTF.models.create_musigma_model(input_shape, **model_kwargs)
    model.load_weights(checkpoint_path)
    if float(tf.__version__[:3]) >2.0:
        pred_mu, pred_var = model.predict([features_test, tf.constant(0, shape=features_test.shape), tf.constant(0, shape=features_test.shape)])
    elif float(tf.__version__[:3]) <=2.0:
        pred_mu, pred_var = model(features_test.astype('float32')).numpy()
    pred_mu=np.ma.array(pred_mu,mask=~mask_test)
    pred_mu=targets_normer.denorm(pred_mu)

    
    fprima=1./targets_normer.b
    if targets_normer.type=="-11": fprima*=2
    test_preds_variance=np.ma.array(pred_var, mask=~mask_test)/(fprima**2)
    #test_preds_variance=np.ma.array(pred_var, mask=~mask_test)*(fprima**2)
    nreas=np.sum(~test_preds_variance.mask,axis=1) 
    test_val_biases_msb_err = (1/nreas)*np.ma.sqrt(np.ma.sum(test_preds_variance, axis=1))


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
    ebarskwargs = {"fmt":'none', "ls":":", 'elinewidth':0.5, 'alpha':0.9}
    print(pred_mu[:,0,0].shape, features_test[:,0,0].shape,test_val_biases_msb_err[:,0].shape )
    plt.errorbar(pred_mu[:,0,0], features_test[:,0,0], xerr=test_val_biases_msb_err[:,0],**ebarskwargs)
    
    plt.plot(pred_mu[:,:,0], features_test[:,:,0], ls="-", color=color_cycle[2], label="Trained with %s"%(model_kwargs['loss_name']), lw=1.5, alpha=0.5)
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
    example_path='point_noise_regression_musigma_%ifeats_%s'%(NFEATS, model_kwargs['loss_name'])
    
    trainingpath = os.path.expanduser(os.path.join(outpath,example_path, "training"))
    make_dir(trainingpath)
    validationpath = os.path.expanduser(os.path.join(outpath,example_path,"validation"))
    make_dir(validationpath)
    
    normerspath = os.path.expanduser(os.path.join(outpath, "data","normers"))
    make_dir(normerspath)
    trainingcat=os.path.join(outpath, "data", "traincat38.pkl")
    trainingvalcat=os.path.join(outpath, "data", "trainvalcat38.pkl")
    validationcat=os.path.join(outpath, "data", "valcat38.pkl")
    testcat=os.path.join(outpath, "data", "testcat.pkl")

    checkpoint_path=os.path.join(trainingpath, "simple_regression.ckpt")

    
    ncases=500
    nreas=1000
    nmsk_obj=5000
    features,targets=makedata(ncases, nreas, f, nmsk_obj, filename=trainingcat)
    features_normer=MLTF.normer.Normer(features, type="01") #sa1
    features=features_normer(features)
    targets_normer=MLTF.normer.Normer(targets, type="01")
    targets=targets_normer(targets)

    features_val,targets_val=makedata(ncases, nreas+100, f, nmsk_obj, filename=trainingvalcat)
    features_val=features_normer(features_val)
    targets_val=targets_normer(targets_val)
    validation_data= ([features_val.data ,targets_val, np.all(~features_val.mask,axis=2,keepdims=True)],None)
    #validation_data= None
    #validation_split=0.3
    validation_split=None
    
    
    logger.info("Data was done")

    train(features,targets, trainingpath, checkpoint_path, reuse=True ,epochs=100000, validation_data=validation_data, validation_split=validation_split, finetune=args.finetune, batch_size=args.batch_size )

    features_test=maketestdata(ncases=100)
    features_test=features_normer(features_test)
    features_val,targets_val=makedata(ncases, nreas, f, nmsk_obj, filename=validationcat)
    features_val=features_normer(features_val)
    targets_val=targets_normer(targets_val)
    validate(features_val, targets_val, checkpoint_path, validationpath, targets_normer )
    test(features, targets,checkpoint_path, f, validationpath, features_test, targets_normer, features_normer)



    
      
if __name__ == "__main__":
    main()
