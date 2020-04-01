#!/bin/bash
#SBATCH -o <output>
#SBATCH -C gpu
#SBATCH -t 10
#SBATCH -c 10
#SBATCH -N <nnodes>
#SBATCH --ntasks-per-node=<ngpu>
#SBATCH --gres=gpu:<ngpu>
#SBATCH -A dasrepo
#SBATCH --requeue
#SBATCH -J pytorch-stsres-cgpu
#SBATCH --exclusive

module load pytorch/v1.4.0-gpu
mkdir -p scaling

# # move data to SSD
# cp -r data /tmp

# run
srun -l python train_multinode.py --ranks_per_node=8 --output_timing='scaling/scaling_multinode.csv' --epochs=2 --log_dir='/tmp/scaling' --data_folder='data' --apex_optim_level=<opt> --batch_size_per_gpu=<bs> --skip_eval --no_use_apex
