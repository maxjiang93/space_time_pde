"""
Utilities for distributed training.
"""

import os
import torch.distributed as dist

def _get_sync_file():
    """Logic for naming sync file using slurm env variables"""
    sync_file_dir = '%s/pytorch-sync-files' % os.environ['SCRATCH']
    os.makedirs(sync_file_dir, exist_ok=True)
    sync_file = 'file://%s/pytorch_sync.%s.%s' % (
        sync_file_dir, os.environ['SLURM_JOB_ID'], os.environ['SLURM_STEP_ID'])
    return sync_file

def init_workers_gloo_file():
    """Initialize workers with GLOO backend and sync file"""
    rank = int(os.environ['SLURM_PROCID'])
    n_ranks = int(os.environ['SLURM_NTASKS'])
    sync_file = _get_sync_file()
    dist.init_process_group(backend='gloo', world_size=n_ranks, rank=rank,
                            init_method=sync_file)
    return rank, n_ranks

def init_workers_nccl_file():
    """Initialize workers with NCCL backend and sync file"""
    rank = int(os.environ['SLURM_PROCID'])
    n_ranks = int(os.environ['SLURM_NTASKS'])
    sync_file = _get_sync_file()
    print('Setting up with sync file', sync_file)
    dist.init_process_group(backend='nccl', world_size=n_ranks, rank=rank,
                            init_method=sync_file)
    return rank, n_ranks

def init_workers_mpi():
    """Initialize workers with MPI backend"""
    dist.init_process_group(backend='mpi')
    rank = dist.get_rank()
    n_ranks = dist.get_world_size()
    return rank, n_ranks

def init_workers(backend=None):
    """Initialize workers for specified backend.

    Note that only a few modes are currently supported:
    - MPI backend
    - NCCL backend with ranks determined by SLURM variables and intialized via
      shared file under $SCRATCH.
    - GLOO backend with rank determined by SLURM variables and intialized via
      shared file under $SCRATCH.
    """
    if backend is None:
        rank, n_ranks = 0, 1
    elif backend == 'mpi':
        rank, n_ranks = init_workers_mpi()
    elif backend == 'nccl':
        rank, n_ranks = init_workers_nccl_file()
    elif backend == 'gloo':
        rank, n_ranks = init_workers_gloo_file()
    return rank, n_ranks