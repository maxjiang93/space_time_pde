## Rayleigh-Bernard 2D Experiment

## Dataset
Below are some instructions for retrieving the sample data. We can generate more data by varying any of the below parameters:
1. Random seed for the initial perturbation. Set to 42 for the example dataset. At a different random seed the solver will produce results simulated under the same set of physics, but different realisations of the same set of equation given different intial perturbations.
2. Raleigh number. Set to 1e6 for the example dataset. Higher Raleigh number leads to stronger turbulence.
3. Temperature of top and bottom plates. Set to +0.5 and -0.5 for the example dataset.

### Retrieving Sample Data
Download the sample simulation data file (~700 MB) from the server
```bash
bash download_data.sh
```

### (Alternatively) Generate new simulation data
Please refer to the Readme page on in the [simulation folder](../../simulation/2d_rayleigh_benard) to generate new data with a different simulation setup.

### Dataset Description
Load and parse the data in python
```python
import numpy as np

filename = 'rb2d_ra1e6_s42.npz'
d = np.load(filename)

print(d.keys())
# >>> ['p', 'b', 'u', 'w', 'bz', 'uz', 'wz', 'write_number', 'sim_time']

print(d['b'].shape)
# >>> (200, 512, 128)
```
Each physical quantity is stored as a (# time steps, # resolution in x, # resolution in z) array.

Below is a description of the variables in this file:
- p: pressure, shape (200, 512, 128)
- b: temperature, shape (200, 512, 128)
- u: velocity in the x direction, shape (200, 512, 128)
- w: velocity in the z direction, shape (200, 512, 128)
- bz: the z derivative of b, shape (200, 512, 128)
- uz: the z derivative of u, shape (200, 512, 128)
- wz: the z derivative of w, shape (200, 512, 128)
- write_number: the sequence index of the simulation frames
- sim_time: simulation time.

## Training
Training is as simple as running the training script. Before training, make sure to mask out the GPU that you want to use.
```bash
# for example running with gpu number 0 and 1.
export CUDA_VISIBLE_DEVICES=0,1  
# run the code in the backend
nohup python train.py --log_dir='log/run1' &> /dev/null & 
# checkout the text logs
tail -f /log/run1/log.txt
# monitor training progress through tensorboard
cd log/run1/tensorboard && tensorboard --logdir . --port 6006
# you may view the tensorboard on http://localhost:6006. Has sample image samples and training curves etc.
```
To view input arguments:
```bash
python train -h
```
The more important arguments are `--alpha_reg` and `--alpha_pde`, which controls the weights between regression loss and pde loss. Setting `--alpha_pde=0` would turn off pde loss during training process.

In order to train the models for the results presented in the paper, you can run 
```bash 
bash run_experiments.sh
```
However, you should have created the corresponding datasets for each experiment from the simulation panel. Refer to the comments in the "run_experiments.sh" script as well.

## Evaluation
To run evaluation for each experiment (and create videos) you can run
```bash
python evaluation.py --eval_dataset='YOUR_EVAL_DATASET' --lres_filter='none' --ckpt='YOUR_CHECKPOINT_DIR/checkpoint_latest.pth.tar_pdenet_best.pth.tar' --save_path='YOUR_LOG_PATH_TO_SAVE_THE_EVAL_RESULTS/' --rayleigh="DATASET's CORRESPONDING RAYLEIGH" --prandtl="DATASET's CORRESPONDING PRANDTL"
```
