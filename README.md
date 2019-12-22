# Physical Constrained Space Time Super-Resolution 

[Methodolgy Slides](https://docs.google.com/presentation/d/13nI5D33ADybplJs5fFD3gM_txTEn0HiaP7gDBHhw6VM/edit#slide=id.g64a817040a_0_73) | [Writeup](https://www.overleaf.com/project/5deacb4f3a2f63000141b1ba) | [Project Folder](https://drive.google.com/open?id=1KybErSl2vU9vfiV_CSO_ApWsypGzezYs)

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
- Test dataloader + U-Net + lig on small data to overfit (@soheil)
- LaTeX writeup (@kamyar)
