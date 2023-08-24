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
import sys, os, glob
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(
        "ignore", category=DeprecationWarning,
        message=r"tostring\(\) is deprecated\. Use tobytes\(\) instead\.")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import MLTF
import numpy as np
import random
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')
import matplotlib.animation as anime
import astropy
import scipy
import scipy.stats
import tensorflow as tf
if float(tf.__version__[:3]) < 2.0:
    print("Using eager execution")
    #tf.compat.v1.enable_eager_execution()
    tf.enable_eager_execution()
    
import logging
logger = logging.getLogger(__name__)



#model_kwargs={'loss_name':'msb', 'use_mask': True, 'hidden_sizes':(5,5), 'activation':'sigmoid', 'layer':MLTF.layer.TfbilacLayer}
model_kwargs={'loss_name':'mse', 'use_mask': True, 'hidden_sizes':(5,), 'activation':'sigmoid', 'layer':tf.keras.layers.Dense}
#model_kwargs={'loss_name':'msb', 'use_mask': True, 'hidden_sizes':(5,5), 'activation':'sigmoid', 'layer':tf.keras.layers.Dense}
NFEATS=2

AUXNAME="deleteme.png"
NFRAMES=100
FIGZISE=(9,4)
stampsize=64

DARKMODE = False # activate or deactivate black and white 

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
    
    print(cp_callback.best)

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
    model=MLTF.models.create_model(input_shape, **model_kwargs)
    model.compile(loss=None, optimizer=opt, metrics = [])

    initial_epoch=0
    dircheck=os.path.dirname(checkpoint_path)
    print(dircheck)
    if os.listdir(dircheck):
        files=glob.glob(os.path.join(dircheck, "*.index"))
        files.sort(key=lambda x: os.path.getmtime(x))
        checkpoint_path=files[-1]
        s=checkpoint_path
        initial_epoch=int(s[s.find("epoch:")+len("epoch:"):s.rfind("_")])
        initial_loss=float(s[s.find("loss:")+len("loss:"):s.rfind(".ckpt.index")])
        print(initial_epoch)
        print(initial_loss)
        cp_callback.best = initial_loss
        if os.path.isfile(checkpoint_path) & reuse:
            logger.info("loading checkpoint weights")
            model.load_weights(checkpoint_path.replace(".index",""))
    

    
    training_data=[features.data, targets, mask]
    
    hist = model.fit(training_data, None, epochs=epochs, verbose=2, 
                     shuffle=True, batch_size=batch_size, initial_epoch=initial_epoch,
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
        hist = model.fit([features, targets, mask], None, epochs=epochs, verbose=2, 
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

    
def linregfunc(x, y, prob=0.68):
	"""
	A linear regression y = m*x + c, with confidence intervals on m and c.
	
	As a safety measure, this function will refuse to work on masked arrays.
	Indeed scipy.stats.linregress() seems to silently disregard masks...
	... and as a safety measure, we compare against scipy.stats.linregress().
	
	"""
	
	if len(x) != len(y):
		raise RuntimeError("Your arrays x and y do not have the same size")
	
	if np.ma.is_masked(x) or np.ma.is_masked(y):
		raise RuntimeError("Do not give me masked arrays")
	
	n = len(x)
	xy = x * y
	xx = x * x
	
	b1 = (xy.mean() - x.mean() * y.mean()) / (xx.mean() - x.mean()**2)
	b0 = y.mean() - b1 * x.mean()
	
	#s2 = 1./n * sum([(y[i] - b0 - b1 * x[i])**2 for i in xrange(n)])
	s2 = np.sum((y - b0 - b1 * x)**2) / n
	
	alpha = 1.0 - prob
	c1 = scipy.stats.chi2.ppf(alpha/2.,n-2)
	c2 = scipy.stats.chi2.ppf(1-alpha/2.,n-2)
	#print 'the confidence interval of s2 is: ',[n*s2/c2,n*s2/c1]
	
	c = -1 * scipy.stats.t.ppf(alpha/2.,n-2)
	bb1 = c * (s2 / ((n-2) * (xx.mean() - (x.mean())**2)))**.5
	#print 'the confidence interval of b1 is: ',[b1-bb1,b1+bb1]
	
	bb0 = c * ((s2 / (n-2)) * (1 + (x.mean())**2 / (xx.mean() - (x.mean())**2)))**.5
	#print 'the confidence interval of b0 is: ',[b0-bb0,b0+bb0]
	
	ret = {"m":b1-1.0, "c":b0, "merr":bb1, "cerr":bb0}
	
	# A little test (for recent numpy, one would use np.isclose() for this !)
	#slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
	'''
	if not abs(slope - b1) <= 1e-6 * abs(slope):
		raise RuntimeError("Slope error, %f, %f" % (slope, b1))
	if not abs(intercept - b0) <= 1e-6 * abs(intercept):
		raise RuntimeError("Intercept error, %f, %f" % (intercept, b0))
	'''
	return ret

def color_plot_ax(ax,x,y,z=None,yerr=None,colorlog=True,xtitle="",ytitle="",ctitle="",ftsize=16,xlim=None, ylim=None,cmap=None,filename=None, colorbar=True,linreg=True):

    if linreg:
        xplot=np.linspace(min(x),max(x))
        #mask=~x.mask&~y.mask
        #x_unmask,y_unmask=x[mask].data,y[mask].data
        are_maskx=type(x)==np.ma.masked_array
        are_masky=type(y)==np.ma.masked_array
        are_masked=(np.ma.is_masked(x))|(np.ma.is_masked(y))|(are_maskx|are_masky)
        if are_masked:
            mask=~x.mask&~y.mask
            x_unmask,y_unmask=x[mask].data,y[mask].data
        else:
            x_unmask,y_unmask=x,y
        
        if yerr is not None:
            ret=linregw(x_unmask,y_unmask,1./(yerr**2))
        else:
            ret=linregfunc(x_unmask,y_unmask)
        m,merr,c, cerr=(ret["m"]+1),ret["merr"],ret["c"],ret["cerr"]
        ax.plot(xplot,m*xplot+c, ls='-',linewidth=2, color='red', label='$\mu_{1}$: %.4f $\pm$ %.4f \n  c$_{1}$: %.4f $\pm$ %.4f'%(m,merr,c, cerr ))
        if DARKMODE: ax.legend(loc='upper left', prop={'size': ftsize-4}, labelcolor="yellow")
        else: ax.legend(loc='upper left', prop={'size': ftsize-4})
            
    if colorlog: 
        c=abs(c)
        colornorm=LogNorm( vmin=np.nanmin(c), vmax=np.nanmax(c))
    else: colornorm=None


    sct=ax.scatter(x, y,c=z, norm=colornorm, marker=".",alpha=0.7,cmap=cmap)

    if yerr is not None:
            if DARKMODE: ebarskwargs = {"fmt":'none', "color":"yellow", "ls":":", 'elinewidth':1.5, 'alpha':1.0}
            else : ebarskwargs = {"fmt":'none', "ls":":", 'elinewidth':1.5, 'alpha':1.0}
            ax.errorbar(x, y, yerr=yerr, **ebarskwargs)

    
    ax.set_xlabel(xtitle, fontsize=ftsize)
    ax.set_ylabel(ytitle, fontsize=ftsize)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    if colorbar:
        cbar=plt.colorbar(sct, ax=ax)
        cbar.ax.set_xlabel(ctitle, fontsize=ftsize-2)
        cbar.ax.xaxis.set_label_coords(0.5,1.1)

    if DARKMODE:
        for pos in ["left","right","top","bottom"]:
            ax.spines[pos].set_color('white')
        ax.yaxis.label.set_color('yellow')
        ax.xaxis.label.set_color('yellow')
        ax.tick_params(axis='x', colors='white',which='both')
        ax.tick_params(axis='y', colors='white',which='both')
        ax.set_facecolor('black')


def make_plot(x1,y1, x2a,y2a, x2b, y2b, func, plotname='test.png', vmin=None, vmax=None,ylim=None,xlim=None, title1="", title2=""):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    import numpy as np

    fig, axs= plt.subplots(1, 2, figsize=FIGZISE)

    norm=None

    
    for ax, title in zip([axs[0],axs[1]],[title1, title2]):
        if DARKMODE:
            ax.set_title(title,size=14, color="yellow")
            ax.patch.set_edgecolor('white')
        else:
            ax.set_title(title,size=14)
        ax.patch.set_linewidth('2') 
        #ax.axis('off')
 
    xlabel=r"$\theta$"
    ylabel=r"$\langle \hat{\theta} - \theta \rangle$"
    color_plot_ax(axs[0],x1,y1,z=None,colorlog=False,xtitle=xlabel,ytitle=ylabel,ctitle="",ftsize=16,xlim=xlim, ylim=ylim,cmap=None,filename=None, colorbar=False,linreg=True)
    #axs[0].set_title(title,size=14, color="yellow")    

    trutheta = np.linspace( -1.0, 2.2, 100)
    trud = func(trutheta)

    axs[1].plot(x2b, y2b, marker=".", color="gray", ls="None", ms=2, label="Training data samples")
    if DARKMODE:
        axs[1].plot(x2a, y2a, ls="-", color="yellow", label="Trained with %s"%(model_kwargs['loss_name']), lw=1.5, alpha=0.5)
        axs[1].plot(trutheta, trud, ls="-", color="white", dashes=(5, 5), lw=2.0, label=r"$d = \sqrt{1 + \theta^2}$")
    else:
        axs[1].plot(x2a, y2a, ls="-", label="Trained with %s"%(model_kwargs['loss_name']), lw=1.5, alpha=0.5)
        axs[1].plot(trutheta, trud, ls="-", color="blue", dashes=(5, 5), lw=2.0, label=r"$d = \sqrt{1 + \theta^2}$")
    
    axs[1].set_xlabel(r"$\theta$ $\mathrm{and}$ $\hat{\theta}$", fontsize=18)
    axs[1].set_ylabel(r"$d$", fontsize=18)
    #axs[1].legend(loc=2, fontsize=12, markerscale=4, numpoints=1, labelcolor="green")
    axs[1].legend(loc=2, fontsize=12, markerscale=4, numpoints=1)
    axs[1].set_xlim(-1.2, 2.4)
    axs[1].set_ylim(0.5, 3.0)

    if DARKMODE: 
        for pos in ["left","right","top","bottom"]:
            axs[1].spines[pos].set_color('white')
        axs[1].yaxis.label.set_color('yellow')
        axs[1].xaxis.label.set_color('yellow')
        axs[1].tick_params(axis='x', colors='white',which='both')
        axs[1].tick_params(axis='y', colors='white',which='both')
        axs[1].set_facecolor('black')
        fig.patch.set_facecolor('black')
    
    fig.tight_layout()
    fig.savefig(plotname, transparent=False) #transparent does not work with gift
    plt.close(fig)

    
def make_animation(features, targets, checkpoint_path, func, valpath, features_test, features_normer, targets_normer):
    fig, ax = plt.subplots(figsize=FIGZISE)
    ims=[]

    MAX_NPOINTS=5000
    mask =np.all(~features.mask,axis=2,keepdims=True)
    mask_test =np.all(~features_test.mask,axis=2,keepdims=True)

    input_shape=features[0].shape #(nreas, nfeas)
    input_shape=(None, features[0].shape[1])

    model=MLTF.models.create_model(input_shape, **model_kwargs )
    
    checkpoints_path=os.path.dirname(checkpoint_path)
    #files = sorted(glob.glob(os.path.join(checkpoints_path, "*.h5")))
    #files = sorted(glob.glob(os.path.join(checkpoints_path, "*.ckpt.index")), key=lambda x: int(x.split('.')[1].replace("-0","")))
    files =glob.glob(os.path.join(checkpoints_path, "*.ckpt.index"))
    epochs=[int(f.rsplit("epoch:",1)[1].rsplit("_loss")[0]) for f in files]
    files=np.array(files)[np.argsort(epochs)]
    checkpoints=["".join(f.rsplit(".index",1)) for f in files]
    sel=NFRAMES//2
    step=len(checkpoints)//sel
    #step=1
    print(files)

    #use last 5 check points
    lastcheck=5
    indxs=[len(checkpoints)-1-i for i in range(lastcheck)]
    if step >0:
        indxs+=[len(checkpoints)-lastcheck -1 -step*i for i in range(sel)]
        aux=np.array(checkpoints)[indxs]
    else:
        aux=np.array(checkpoints)
    iterlist= np.flip(aux).tolist() +aux.tolist()
    for f in iterlist:
        status=model.load_weights(f)
        #status.assert_consumed()
        status.expect_partial()

        if float(tf.__version__[:3]) >2.0:
            preds = model([features, tf.constant(0, shape=(features.shape[0], 1 , 1)), tf.constant(0, shape=features.shape)]).numpy()
            #preds = model.predict([features, tf.constant(0, shape=(features.shape[0], 1 , 1)), tf.constant(0, shape=features.shape)])
        elif float(tf.__version__[:3]) <2.0:
            preds = model(features.astype('float32')).numpy()
        preds=np.ma.array(preds,mask=~mask)

        val_biases_msb = np.mean(preds, axis=1, keepdims=True) - targets

        loss_val_direct=np.ma.mean(np.square(val_biases_msb))    
        loss_val=MLTF.loss_functions.msb(tf.convert_to_tensor(targets.astype('float32')), tf.convert_to_tensor(preds.astype('float32')), mask=tf.convert_to_tensor((mask*1.0).astype('float32')))

        loss_val_file=float(f.rsplit("_loss:",1)[1].rsplit(".ckpt")[0])
        epoch=int(f.rsplit("epoch:",1)[1].rsplit("_loss")[0])
        print(loss_val_direct,loss_val.numpy(),loss_val_file  )

        x1=np.ma.array(targets_normer.denorm(targets)[:,0,0],mask=False)
        y1=val_biases_msb[:,0,0]

        if float(tf.__version__[:3]) >2.0:
            #test_preds = model.predict([features_test, tf.constant(0, shape=features_test.shape), tf.constant(0, shape=features_test.shape)])
            test_preds = model([features_test, tf.constant(0, shape=features_test.shape), tf.constant(0, shape=features_test.shape)])
        elif float(tf.__version__[:3]) <=2.0:
            test_preds = model(features_test.astype('float32')).numpy()
        test_preds=np.ma.array(test_preds,mask=~mask_test)
        x2a=targets_normer.denorm(test_preds)[:,:,0]
        y2a=features_normer.denorm(features_test)[:,:,0]

        #Selecting few training data points for the plot
        npoints=10000
        targets_1d = np.ma.concatenate(targets_normer.denorm(targets).T)[0]
        features_1d = np.ma.concatenate(features_normer.denorm(features)[:,:,0].T)
        showtrainindices = np.arange(targets_1d.size)
        np.random.shuffle(showtrainindices)
        showtrainindices = showtrainindices[:npoints]
        x2b= targets_1d[showtrainindices]
        y2b =features_1d[showtrainindices]

        filename=os.path.join(valpath, AUXNAME)
        make_plot(x1,y1, x2a,y2a,x2b,y2b, func, plotname=filename, ylim=[-0.04,0.05], title1="%s: %.2e"%(model_kwargs["loss_name"].upper(),loss_val_file), title2="Epoch: %i"%(epoch) )#, vmin=0, vmax=0.015, ylim=[-0.1,0.1])
        im = ax.imshow(plt.imread(filename), animated = True)
        ims.append([im])

        
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.axis('off')
    ani = anime.ArtistAnimation(fig, ims, interval=300, blit=True)
    filename=os.path.join(valpath, 'inverse_regression.gif')
    ani.save(filename)
    plt.close(fig)
    
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
    example_path='point_noise_regression_animation_%ifeats_%s'%(NFEATS, model_kwargs['loss_name'])
    
    trainingpath = os.path.expanduser(os.path.join(outpath,example_path, "training"))
    make_dir(trainingpath)
    checkpointpath=os.path.join(trainingpath, "checkpoints")
    make_dir(checkpointpath)
    validationpath = os.path.expanduser(os.path.join(outpath,example_path,"validation"))
    make_dir(validationpath)
    
    normerspath = os.path.expanduser(os.path.join(outpath, "data","normers"))
    make_dir(normerspath)
    trainingcat=os.path.join(outpath, "data", "traincat.pkl")
    trainingvalcat=os.path.join(outpath, "data", "trainvalcat.pkl")
    validationcat=os.path.join(outpath, "data", "valcat.pkl")
    testcat=os.path.join(outpath, "data", "testcat.pkl")

    checkpoint_path=os.path.join(checkpointpath, "epoch:{epoch:1d}_loss:{loss:.3e}.ckpt")
    #checkpoint_path=os.path.join(checkpointpath, "simple_regression.{epoch:02d}-{loss:.10f}.h5")

    
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
    features_test=maketestdata(ncases=100)
    features_test=features_normer(features_test)
    
    logger.info("Data was done")

    train(features,targets, trainingpath, checkpoint_path, reuse=True, epochs=3025, validation_data=validation_data, validation_split=validation_split, finetune=args.finetune, batch_size=args.batch_size )

    features_val,targets_val=makedata(ncases, nreas, f, nmsk_obj, filename=validationcat)
    features_val=features_normer(features_val)
    targets_val=targets_normer(targets_val)
    

    make_animation(features_val, targets_val, checkpoint_path, f, validationpath, features_test, features_normer, targets_normer)



    
      
if __name__ == "__main__":
    main()
