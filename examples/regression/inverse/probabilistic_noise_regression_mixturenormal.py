import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(
        "ignore", category=DeprecationWarning,
        message=r"tostring\(\) is deprecated\. Use tobytes\(\) instead\.")


import sys, os
sys.path.append("../..")
import SHE_KerasML.python.SHE_KerasML as SHE_KerasML
import numpy as np
import random
import json

import matplotlib
import astropy
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
if float(tf.__version__[:3]) < 2.0:
    print("Using eager execution")
    tf.compat.v1.enable_eager_execution()
    #tf.enable_eager_execution()
    
import logging
logger = logging.getLogger(__name__)


model_kwargs={'loss_name':'nll', 'use_mask': True, 'hidden_sizes':(5,5), 'activation':'sigmoid', 'layer':SHE_KerasML.layer.TfbilacLayer, "ncomp":1}
NFEATS=1

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Main script for training and validating fiducial simulations')
    parser.add_argument('--workdir',
                        default='testing/out/probabilistic_regression_mixturenormal_%ifeats_%s'%(NFEATS, model_kwargs['loss_name']), 
                        help='diractory of work')
    parser.add_argument('--train', default=False,
                        action='store_const', const=True, help='Train point estimate')
    parser.add_argument('--validate_', default=False,
                        action='store_const', const=True, help='Validate point estimate')
    parser.add_argument('--finetune', default=False,
                        action='store_const', const=True, help='Use SGD for getting as low as possible in the converged region')
   
    args = parser.parse_args()

    return args




def noise(n): return np.random.randn(n)
def f(x): return np.sqrt(1.0 + x**2)
def g(x): return x**3

def makedata(ncases, nreas, func, nmsk_obj=0, noise_scale=0.1, theta_min=0.25 , theta_max=2.0):
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

    logger.info("Data was done")
        
    return features, targets

def train(features, targets, trainpath, checkpoint_path=None, reuse=True, finetune=False, epochs=1000 ):

    mask =np.all(~features.mask,axis=2,keepdims=True)
    caseweights=None
    
    #features_normer=SHE_KerasML.normer.Normer(features, type=inputtype)
    #features=features_normer(features)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                     monitor='loss',
                                                     save_weights_only=True,
                                                     save_best_only= True,
                                                     verbose=1, 
                                                     save_freq='epoch')


    input_shape=features[0].shape #(nreas, nfeas)
    opt=tf.keras.optimizers.Adam(learning_rate=0.01)
    model=SHE_KerasML.model.create_probabilistic_model_mixturenormal(input_shape, **model_kwargs)
    model.compile(loss=None, optimizer=opt, metrics = [])
    if os.path.isfile(checkpoint_path+'.index') & reuse:
        logger.info("loading checkpoint weights: %s"%(checkpoint_path))
        model.load_weights(checkpoint_path)
    
    model.compile(loss=None, optimizer=opt, metrics = [])
    hist = model.fit([features, targets, mask], None, epochs=epochs, verbose=2, 
                      shuffle=True, batch_size=None, 
                      callbacks=[cp_callback])

    history_path=os.path.join(trainpath, "history.txt")
    data=[]
    if reuse:
        if os.path.isfile(history_path):
            logger.info("loading history %s"%(history_path))
            with open(history_path) as f:
                data = json.load(f)
        else:
            logger.info("%s is not a file"%(history_path))
            
    history_list= data+ hist.history['loss']
    with open(history_path, 'w') as outfile:
        json.dump(history_list, outfile)

    historyimg=os.path.join(trainpath, "history.png")
    SHE_KerasML.plot.plot_history(history_path, historyimg, ylog=False)

    
    logger.info("***--- TRAINING FINISHED --- ***")

    

def validate(features, targets, checkpoint_path, valpath):
    
    MAX_NPOINTS=5000
    
    input_shape=features[0].shape #(nreas, nfeas)
    model=SHE_KerasML.model.create_probabilistic_model_mixturenormal(input_shape, **model_kwargs )

    model.load_weights(checkpoint_path)

    if float(tf.__version__[:3]) >2.0:
        preds = model([features, tf.constant(0, shape=(features.shape[0], 1 , 1)), tf.constant(0, shape=features.shape)]).mean().numpy()
        preds_variance = model([features, tf.constant(0, shape=(features.shape[0], 1 , 1)), tf.constant(0, shape=features.shape)]).variance().numpy()
        
    elif float(tf.__version__[:3]) <2.0:
        preds = model(features.astype('float32')).mean().numpy()
        preds_variance = model(features.astype('float32')).variance().numpy()

    print(preds.shape)
    print(preds_variance.shape)
        
    val_biases_msb = np.mean(preds, axis=1) - targets[:,0,0]
    nreas=preds_variance.shape[1]
    val_biases_msb_err = (1/nreas)*np.sqrt(np.sum(preds_variance, axis=1))

    #val_biases_msb_err=None
        
    filename=os.path.join(valpath, "bias_vs_targets.png")
    SHE_KerasML.plot.color_plot(targets[:,0,0],val_biases_msb, None ,False, r"$\theta$" ,r"$\langle \hat{\theta} - \theta \rangle$", "" , yerr=val_biases_msb_err, title="", ftsize=18,cmap="gnuplot", filename=filename, npoints_plot=MAX_NPOINTS, linreg=True, alpha_err=1.0)
    

        
    logger.info("***--- VALIDATING FINISHED --- ***")


def test(features, targets, checkpoint_path, func, path):
    
    

    #Selecting few training data points for the plot
    npoints=10000
    targets_1d = np.concatenate(targets.T)[0]
    features_1d = np.concatenate(features[:,:,0].T)
    showtrainindices = np.arange(targets_1d.size)
    np.random.shuffle(showtrainindices)
    showtrainindices = showtrainindices[:npoints]
    targets_1d= targets_1d[showtrainindices]
    features_1d = features_1d[showtrainindices]


    
    test_ncase=100
    func_val=np.linspace( 0.5, 3,test_ncase)
    if NFEATS==2:
        features_test = np.transpose(np.array([[ func_val, np.array([0]*test_ncase) ] for rea in range(1)]),axes=(2,0,1))
    elif NFEATS==1:
        features_test = np.transpose(np.array([[ func_val ] for rea in range(1)]),axes=(2,0,1))
    
    #Loading model
    input_shape=features_test[0].shape #(nreas, nfeas)
    model=SHE_KerasML.model.create_probabilistic_model_mixturenormal(input_shape, **model_kwargs)
    model.load_weights(checkpoint_path)
    if float(tf.__version__[:3]) >2.0:
        test_preds = model.predict([features_test, tf.constant(0, shape=features_test.shape), tf.constant(0, shape=features_test.shape)]).mean().numpy()
        test_preds_std = model.predict([features_test, tf.constant(0, shape=features_test.shape), tf.constant(0, shape=features_test.shape)]).stddev().numpy()
    elif float(tf.__version__[:3]) <=2.0:
        test_preds = model(features_test.astype('float32')).mean().numpy()
        test_preds_std = model(features_test.astype('float32')).stddev().numpy()

    # True function    
    trutheta = np.linspace( -1.0, 2.2, 100)
    trud = func(trutheta)
    color_cycle = ["#1b9e77", "#d95f02", "#7570b3"]

    plt.plot(targets_1d, features_1d, marker=".", color="gray", ls="None", ms=2, label="Training data samples")
    ebarskwargs = {"fmt":'none', "color":color_cycle[2], "ls":":", 'elinewidth':0.5, 'alpha':1.0}
    plt.errorbar(test_preds[:,0], features_test[:,0,0], xerr=test_preds_std[:,0],**ebarskwargs)
    plt.plot(test_preds[:,0], features_test[:,:,0], ls="-", color=color_cycle[2], label="Trained with %s"%(model_kwargs['loss_name']), lw=1.5, alpha=0.5)
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
    trainingpath = os.path.expanduser(os.path.join(outpath,"training"))
    make_dir(trainingpath)
    validationpath = os.path.expanduser(os.path.join(outpath,"validation"))
    make_dir(validationpath)
    normerspath = os.path.expanduser(os.path.join(trainingpath ,"normers"))
    make_dir(normerspath)
    checkpoint_path=os.path.join(trainingpath, "simple_regression.ckpt")


    ncases=500
    nreas=100
    nmsk_obj=5000
    features,targets=makedata(ncases, nreas, f, nmsk_obj)
    logger.info("Data was done")

    train(features, targets, trainingpath, checkpoint_path, epochs=100 )

    features_val,targets_val=makedata(ncases, nreas, f, nmsk_obj)

    validate(features_val, targets_val, checkpoint_path, validationpath )

    test(features, targets,checkpoint_path, f, validationpath)
      
if __name__ == "__main__":
    main()
