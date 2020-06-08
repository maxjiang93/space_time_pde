#!/bin/bash  

# set the name of datasets you would like to use for traning/validation
train_dataset_name = "rb2d_ra1e6_s42.npz"
eval_dataset_name  = "rb2d_ra1e6_s42.npz"

# Rayleigh and Prandtl numbers - set according to your dataset
rayleigh=1000000
prandtl=1
gamma=0.0125
log_dir_name="./log/Exp1/"
mkdir -p $log_dir_name
CUDA_VISIBLE_DEVICES=0 python train.py --epochs=100 --data_folder=data --log_dir=$log_dir_name --alpha_pde=$gamma --train_data=$train_dataset_name --eval_data=$eval_dataset_name --rayleigh=$rayleigh --prandtl=$prandtl

# please see the train.py for further tunable arguments during the training process