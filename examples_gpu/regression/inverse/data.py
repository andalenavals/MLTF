import os, sys
import numpy as np
import pickle
import logging
import random
logger = logging.getLogger(__name__)

def noise(n): return np.random.randn(n)

def makedata(ncases, nreas, func, nmsk_obj=0, noise_scale=0.1, theta_min=0.25 , theta_max=2.0, nfeats=2, filename=None, shuffle=True, ):

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
        if nfeats==2:
            aux.append([func(thetas)+nos, nos])
        if nfeats==1:
            aux.append([func(thetas)+nos])
    features=np.array(aux)
    features= np.transpose(features,axes=(2,0,1))

    # Mask
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

def maketestdata(ncases=100, nfeats=2):
    func_val=np.linspace( 0.5, 3,ncases)
    if nfeats==2:
        features_test = np.transpose(np.array([[ func_val, np.array([0]*ncases) ] for rea in range(1)]),axes=(2,0,1))
    elif nfeats==1:
        features_test = np.transpose(np.array([[ func_val ] for rea in range(1)]),axes=(2,0,1))
    return features_test
   
