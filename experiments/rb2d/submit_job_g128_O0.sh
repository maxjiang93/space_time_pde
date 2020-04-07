#!/bin/bash
#SBATCH -o .slurm_logs/ngpu_g128_O0.out
#SBATCH -C gpu
#SBATCH -t 20
#SBATCH -c 10
#SBATCH -N 16
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH -A dasrepo
#SBATCH --requeue
#SBATCH -J pytorch-stsres-cgpu
#SBATCH --exclusive

module load pytorch/v1.4.0-gpu
mkdir -p scaling

# # move data to SSD
# cp -r data /tmp

# run (lr=2e-2)
srun -l python train_multinode.py --ranks_per_node=8 --output_timing='scaling/scaling_multinode.csv' --epochs=100 --data_folder='data' --apex_optim_level=O0 --batch_size_per_gpu=14 --skip_eval --use_apex --lr=2e-2 --log_interval=1 --alpha_pde 0 --lr_scheduler --lr_ramp --clip_grad=10000.  --log_dir='log/scaling128gpu_lr2e-2'

# run (lr=4e-2)
srun -l python train_multinode.py --ranks_per_node=8 --output_timing='scaling/scaling_multinode.csv' --epochs=100 --data_folder='data' --apex_optim_level=O0 --batch_size_per_gpu=14 --skip_eval --use_apex --lr=4e-2 --log_interval=1 --alpha_pde 0 --lr_scheduler --lr_ramp --clip_grad=10000.  --log_dir='log/scaling128gpu_lr4e-2'

# run (lr=8e-2)
srun -l python train_multinode.py --ranks_per_node=8 --output_timing='scaling/scaling_multinode.csv' --epochs=100 --data_folder='data' --apex_optim_level=O0 --batch_size_per_gpu=14 --skip_eval --use_apex --lr=8e-2 --log_interval=1 --alpha_pde 0 --lr_scheduler --lr_ramp --clip_grad=10000.  --log_dir='log/scaling128gpu_lr8e-2'
