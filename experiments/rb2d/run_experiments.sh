#!/bin/bash  


############################################
############################################
# Exp1 - training

# Table I's Experiments
train_dataset_name="train_data_placeholder.npz"
eval__dataset_name="train_data_placeholder.npz"
# Rayleigh and Prandtl numbers - set according to your dataset
rayleigh=1000000
prandtl=1

for gamma in 0 0.0125 0.025 0.05 0.1 0.2 0.4 0.8 1 
do  
    echo running Exp1-gamma=$gamma
    log_dir_name="./log/Exp1/exp_gamma=""$gamma"
    mkdir -p $log_dir_name
    CUDA_VISIBLE_DEVICES=0 python train.py --epochs=100 --data_folder=data --log_dir=$log_dir_name --alpha_pde=$gamma --train_data=$train_dataset_name --eval_data=$eval__dataset_name --rayleigh=$rayleigh --prandtl=$prandtl
done

############################################
############################################
# Exp2 - training

# Table II's Experiments
train_dataset_name="train_data_placeholder.npz"
eval__dataset_name="train_data_placeholder.npz"

# Baseline (II)
log_dir_name="./log/Exp2/exp_baseline"
mkdir -p $log_dir_name
echo running Exp2-BaselineII
CUDA_VISIBLE_DEVICES=0 python train_baseline.py --epochs=100 --data_folder=data --log_dir=$log_dir_name --train_data=$train_dataset_name --eval_data=$eval__dataset_name

# MeshFreeFlowNet, gamma = 0, 0.0125
# Rayleigh and Prandtl numbers - set according to your dataset
rayleigh=1000000
prandtl=1
for gamma in 0 0.0125
do
    echo running Exp2-gamma=$gamma
    log_dir_name="./log/Exp2/exp_gamma=""$gamma"
    mkdir -p $log_dir_name
    CUDA_VISIBLE_DEVICES=0 python train.py --epochs=100 --data_folder=data --log_dir=$log_dir_name --alpha_pde=$gamma --train_data=$train_dataset_name --eval_data=$eval__dataset_name --rayleigh=$rayleigh --prandtl=$prandtl
done
############################################
############################################
# Exp3 - training

# Table III's Experiments
train_10datasets_name="train_10datasets_placeholder.npz"
eval__10datasets_name="train_10datasets_placeholder.npz"
gamma=0.0125
# Rayleigh and Prandtl numbers - set according to your dataset
rayleigh=1000000
prandtl=1

log_dir_name="./log/Exp3/exp_baseline_gamma=""$gamma"
echo running Exp3-10InitialConditions
mkdir -p $log_dir_name
CUDA_VISIBLE_DEVICES=0 python train.py --epochs=1000 --data_folder=data --log_dir=$log_dir_name --alpha_pde=$gamma --train_data=$train_10datasets_name --eval_data=$eval__10datasets_name --rayleigh=$rayleigh --prandtl=$prandtl

############################################
############################################ss
# Exp4 - training

### NOTE: 
### you should run the below experiment with 10 boundary conditions, i.e. (rayleigh & prandtl) ...
### ...where for each (rayleigh & prandtl) pair the corresponding train and eval data are used

# Table IV's Experiments
train_dataset_name="train_data_placeholder.npz"
eval__dataset_name="train_data_placeholder.npz"
gamma=0.0125
# Rayleigh and Prandtl numbers - set according to your dataset
rayleigh=1000000
prandtl=1

log_dir_name="./log/Exp4/exp_baseline_gamma=""$gamma"
echo running Exp4-10BoundaryConditions
mkdir -p $log_dir_name
CUDA_VISIBLE_DEVICES=0 python train.py --epochs=100 --data_folder=data --log_dir=$log_dir_name --alpha_pde=$gamma --train_data=$train_dataset_name --eval_data=$eval__dataset_name --rayleigh=$rayleigh --prandtl=$prandtl
