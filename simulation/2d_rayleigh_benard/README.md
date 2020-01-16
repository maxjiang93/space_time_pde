## Dedalus Installation Instructions
Please follow the following steps for installing Dedalus. Modifed instructions from [here](https://dedalus-project.readthedocs.io/en/latest/installation.html).

Before starting these steps, make sure that you have a working conda installation. If not follow the instructions on the [conda installation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) page.
```bash
# Download and run the official installation script.
wget https://raw.githubusercontent.com/DedalusProject/conda_dedalus/master/install_conda.sh
bash conda_install.sh
```

## Create the simulation file for the ML pipeline
Follow the following steps to create a simulation
```bash
# Make sure dedalus environment is activated
conda activate dedalus
# Check the arguments / simulation parameters.
python rayleigh_bernard.py -h
# Run with 4 mpi processes (additional flags can be added to change simulation params.)
mpiexec -n 4 python3 rayleigh_benard.py
# Merge processes
python -m dedalus merge_procs snapshots
# Convert into npz file to be consumed by the machine learning pipeline
python convert_to_npz.py -f 'snapshots/snapshots_s*.h5' -o 'rb2d_ra1e6_s42.npz'
```

## Create a video of the simulation
```bash
# Generate png frames
python plot_slices.py snapshots/*.h5
# Stich png frames into video
bash create_video.sh
```
