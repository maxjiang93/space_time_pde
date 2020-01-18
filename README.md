# Physical Constrained Space Time Super-Resolution 

[Progress Update Slides](https://docs.google.com/presentation/d/1YODV57luQjzG2T7SCabdBX0pEg6VO_T7K44V8-K6sIA/edit?usp=sharing) | [Methodolgy Slides](https://docs.google.com/presentation/d/13nI5D33ADybplJs5fFD3gM_txTEn0HiaP7gDBHhw6VM/edit#slide=id.g64a817040a_0_73) | [Writeup](https://www.overleaf.com/project/5deacb4f3a2f63000141b1ba) | [Project Folder](https://drive.google.com/open?id=1KybErSl2vU9vfiV_CSO_ApWsypGzezYs)

This is the code repository for the physical constrained space time super-resolution project.

## Organization
Here is a rough proposed organization of the code repo. Feel free to add additional folders / files to this list with a logical organization.
```bash
├── doc
│   ├── pde_constraints.png
│   └── pde_layer_schematic.png
├── experiments
│   └── rb2d
│       ├── dataloader.py
│       ├── model.py
│       ├── README.md
│       └── train.py
├── README.md
├── simulation
│   └── 2d_rayleigh_benard
│       ├── convert_to_npz.py
│       ├── create_video.sh
│       ├── plot_slices.py
│       └── rayleigh_benard.py
└── src
    ├── implicit_net.py
    ├── implicit_net_test.py
    ├── local_implicit_grid_integration_test.py
    ├── local_implicit_grid.py
    ├── local_implicit_grid_test.py
    ├── metrics.py
    ├── pde.py
    ├── pde_test.py
    ├── README.md
    ├── regular_nd_grid_interpolation.py
    ├── regular_nd_grid_interpolation_test.py
    ├── unet.py
    └── utils.py
```

## TODO
- ~~Setup repo and write organization doc (@max)~~
- ~~Add migrate LearnableVoxelGrid to lig and add temporal aspects into the framework (@max)~~
- ~~Implement PDE constraints layer (@max)~~
- ~~Convert data samples to `.npy` format before 12/10 and hand over to Soheil for testing (@max)~~
- ~~Add detailed writeup and examples on how to use the PDE layers (@max)~~
- ~~Create data loader (@soheil)~~
- ~~Create U-Net (@soheil)~~
- ~~Test dataloader + U-Net + lig on small data to overfit (@max)~~
- ~~Create and debug training and evaluation script (@max)~~
- ~~Add tensorboard to track training progress (@max)~~
- ~~Create videos with evaluation (@max)~~
- Create simulation data (similar b.c., different init. (initialization seeds)) (@soheil)
- Rayleigh-Benard Training Cases (@soheil)
    - Train on __one__ simulation and test on __same__ b.c. & a __different__ init. (Effect of init. (I)) 
    - Train on __multiple__ simulations (__same__ b.c. & __different__ inits.). Test on __same__ b.c. & a __different__ init. (Effect of init. (II)). 
    - Train on __different__ b.c. & __different__ inits. Test on a __different__ b.c. & a random init. (Effect of b.c. & init.). 
- LaTeX writeup (@kamyar @soheil)
