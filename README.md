# Physical Constrained Space Time Super-Resolution 

[Methodolgy Slides](https://docs.google.com/presentation/d/13nI5D33ADybplJs5fFD3gM_txTEn0HiaP7gDBHhw6VM/edit#slide=id.g64a817040a_0_73) | [Writeup](https://www.overleaf.com/project/5deacb4f3a2f63000141b1ba) | [Project Folder](https://drive.google.com/open?id=1KybErSl2vU9vfiV_CSO_ApWsypGzezYs)

This is the code repository for the physical constrained space time super-resolution project.

## Organization
Here is a rough proposed organization of the code repo. Feel free to add additional folders / files to this list with a logical organization.
```bash
- srcs
  - unet.py          # unet model and utilities
  - lig.py           # learnable implicit grid layer
  - utils.py         # utility functions
  - pde.py           # pde loss layer
  - metrics.py       # metrics for evaluating performance
- experiments
  - rb2d             # 2d rayleigh-benard experiment
    - dataloader.py  # pytorch dataloader for loading data
    - model.py       # model code
    - train.py       # main code for training the model
```

## TODO
- ~~Setup repo and write organization doc (@max)~~
- Implement PDE constraints layer (@max)
- Convert data samples to `.npy` format before 12/10 and hand over to Soheil for testing (@max)
- Create data loader (@soheil)
- Create U-Net (@soheil)
- Test dataloader + U-Net + lig on small data to overfit (@soheil)
- LaTeX writeup (@kamyar)
