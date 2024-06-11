import os, glob
import MLTF
import numpy as np
from astropy.io import fits
import pickle




import logging
logger = logging.getLogger(__name__)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Main script for training and validating fiducial simulations')
 
    parser.add_argument('--cat',
                        default=None, 
                        help='Catalog')

    parser.add_argument('--rad_model_config',
                        default=None, 
                        help='Config file containing network properties of trurad model (apply smart size a first selection)')
    parser.add_argument('--rad_normer_feats',
                        default=None, 
                        help='normed used for the training of truerad feats')
    parser.add_argument('--rad_normer_targets',
                        default=None, 
                        help='normed used for the training of truerad targets')
    parser.add_argument('--rad_features',
                        default=[],
                        nargs='+',
                        help='Features for training mu')
    
 
    parser.add_argument('--g_features',
                        default=[],
                        nargs='+',
                        help='Features for training g')
    parser.add_argument('--g_model_config',
                        default=None, 
                        help='Config file containing network properties of g model (we need g model to train weights)')
    parser.add_argument('--g_normer',
                        default=None, 
                        help='normed used for the training of g')

    args = parser.parse_args()
    return args


def get_shear_preds(cat, shearfiles):
    [shear_model_config, shear_feats_normer_name, inputlabels]=shearfiles
    with open(shear_feats_normer_name, 'rb') as handle: shear_feats_normer = pickle.load(handle)
    features=np.array([np.array([cat[f] for f in inputlabels]).T])
    point_modelkwargs=MLTF.tools_momentsml.get_modelkwargs(configfile=shear_model_config, input_shape=features[0].shape, use_mask=True, noutputs=2)

    weights=[os.path.join(wfold, "weights.ckpt") for wfold in glob.glob("%s/**/"%(os.path.dirname(shear_model_config)))]
    shear_preds=MLTF.tools_momentsml.getpreds_model(shear_feats_normer(features).astype('float32'),
                                                    modelkwargs= point_modelkwargs, model_weights=weights,
                                                    useprocess=True, combine="mean", concatenatenn=True)
    return shear_preds[0,:,:]

def get_rad_preds(cat, radfiles):
    [rad_model_config, rad_feats_names, rad_targets_names,inputlabels]=radfiles
    with open(rad_feats_names, 'rb') as handle: rad_features_normer = pickle.load(handle)
    with open(rad_targets_names, 'rb') as handle: rad_targets_normer = pickle.load(handle)

    features=np.array([np.array([cat[f] for f in inputlabels]).T])
    point_modelkwargs=MLTF.tools_momentsml.get_modelkwargs(configfile=rad_model_config, input_shape=features[0].shape, use_mask=True, noutputs=1)
    
    weights=[os.path.join(wfold, "weights.ckpt") for wfold in glob.glob("%s/**/"%(os.path.dirname(rad_model_config)))]
    rad_preds=MLTF.tools_momentsml.getpreds_model(rad_features_normer(features).astype('float32'), modelkwargs= point_modelkwargs, model_weights=weights, useprocess=True, combine="mean")

    rad_preds=rad_targets_normer.denorm(rad_preds)
    return rad_preds[0,:,0]
   
       
  
def main():    
    args = parse_args()
    loggerformat='PID %(process)06d | %(asctime)s | %(levelname)s: %(name)s(%(funcName)s): %(message)s'
    #logging.basicConfig(format=loggerformat, level=logging.DEBUG)
    logging.basicConfig(format=loggerformat, level=logging.INFO)
    #logging.basicConfig(format=loggerformat, level=logging.NOTSET)
    


    assert os.path.isfile(args.cat)
    with fits.open(args.cat) as hdul:
        cat=hdul[1].data
        columns = hdul[1].columns
    
    radfiles=[args.rad_model_config, args.rad_normer_feats, args.rad_normer_targets, args.rad_features]
    rad_preds=get_rad_preds(cat, radfiles)
    
    shearfiles=[args.g_model_config, args.g_normer,  args.g_features]
    shearpreds=get_shear_preds(cat, shearfiles)

 
    for arr,colname in zip([rad_preds, shearpreds[:,0], shearpreds[:,1]],["pred_rad", "pred_s1", "pred_s2"]):
        if colname not in columns.names:
            columns+=fits.Column(name=colname, format='D', array=arr)
        
    hdu = fits.BinTableHDU.from_columns(fits.ColDefs(columns))
    for arr,colname in zip([rad_preds, shearpreds[:,0], shearpreds[:,1]],["pred_rad", "pred_s1", "pred_s2"]):
        hdu.data[colname] = arr
    hdu.writeto(args.cat, overwrite=True)
  
  
            
if __name__ == "__main__":
    main()
