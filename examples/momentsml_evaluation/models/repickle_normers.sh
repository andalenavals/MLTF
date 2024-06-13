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


FILES=($RAD_NORMER_FEATURES $RAD_NORMER_TARGET $G_NORMER)



cd $START_PATH

python $COMMAND --files ${FILES[@]} 
