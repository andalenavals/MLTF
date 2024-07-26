#!/bin/bash

START_PATH=/users/aanavarroa/original_gitrepos/MLTF/examples/momentsml_evaluation/models
COMMAND=$START_PATH/changenormerpickle.py

TRAINING_MOM=/users/aanavarroa/original_gitrepos/MLTF/examples/momentsml_evaluation/models
#############################################################
###################### SIZE ESTIMATOR #######################
#############################################################
RADDIR=truhlr-trucat-nw-ss_10nodes_varback_varpsf_mse_bulgedisk_1e5_snr7_higmom_samereas/training/
RAD_NORMER_FEATURES=$TRAINING_MOM/$RADDIR/normers/trurad_input_adamom_g1adamom_g2adamom_sigmaadamom_fluxadamom_rho4skymadpsf_adamom_g1psf_adamom_g2psf_adamom_sigmapsf_adamom_rho4psf_mom_rho4psf_mom_M4_1psf_mom_M4_2_normer.pkl
RAD_NORMER_TARGET=$TRAINING_MOM/$RADDIR/normers/trurad_targets_normer.pkl

#############################################################
###################### POINT ESTIMATOR ######################
#############################################################
GDIR=/g1g2ind-grid-trucat-nw-ss_24nodes_varback_varpsf_psfadamom_nosel_lr0.001_samereas_flagship_1e4/training/
G_NORMER=$TRAINING_MOM/$GDIR/normers/tp_input_adamom_g1adamom_g2adamom_sigmaadamom_fluxadamom_rho4skymadpsf_adamom_g1psf_adamom_g2psf_adamom_sigmapsf_adamom_rho4psf_mom_rho4psf_mom_M4_1psf_mom_M4_2_normer.pkl

#############################################################
################# WEIGHT ESTIMATOR (ws)######################
##############################################################
WSDIR=/w-blended-sexcat-w-ss_varback_varpsf_10nodes_fracpixmd20_lr0.01_snr7prdhlr02_mswbmudot/training/
WS_NORMER=$TRAINING_MOM/$WSDIR/normers/tw_input_adamom_g1adamom_g2adamom_sigmaadamom_fluxadamom_rho4skymadpsf_adamom_g1psf_adamom_g2psf_adamom_sigmapsf_adamom_rho4fracpix_md20_normer.pkl

##############################################################
####################### WM ESTIMATOR #########################
##############################################################
WMDIR=mw-blended-sexcat-w-ss_varpsf_varback_10x10elu_1step_fracpixmd20_snr7predhlr02_lr0.0001_mswcbmudot/training/
M_NORMER=$TRAINING_MOM/$WMDIR/normers/tw-blended-imgsize_input_m_fracpix_md20_normer.pkl
W_NORMER=$TRAINING_MOM/$WMDIR/normers/tw-blended-imgsize_input_w_adamom_g1adamom_g2adamom_sigmaadamom_fluxadamom_rho4skymadpsf_adamom_g1psf_adamom_g2psf_adamom_sigmapsf_adamom_rho4_normer.pkl


FILES=($RAD_NORMER_FEATURES $RAD_NORMER_TARGET $G_NORMER $WS_NORMER $W_NORMER $M_NORMER)



cd $START_PATH

python $COMMAND --files ${FILES[@]} 
