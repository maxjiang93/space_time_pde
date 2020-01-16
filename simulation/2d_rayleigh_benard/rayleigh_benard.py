"""
Dedalus script for 2D Rayleigh-Benard convection.

This script uses a Fourier basis in the x direction with periodic boundary
conditions.  The equations are scaled in units of the buoyancy time (Fr = 1).

This script can be ran serially or in parallel, and uses the built-in analysis
framework to save data snapshots in HDF5 files.  The `merge_procs` command can
be used to merge distributed analysis sets from parallel runs, and the
`plot_slices.py` script can be used to plot the snapshots.

To run, merge, and plot using 4 processes, for instance, you could use:
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 -m dedalus merge_procs snapshots
    $ mpiexec -n 4 python3 plot_slices.py snapshots/*.h5

This script can restart the simulation from the last save of the original
output to extend the integration.  This requires that the output files from
the original simulation are merged, and the last is symlinked or copied to
`restart.h5`.

To run the original example and the restart, you could use:
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 -m dedalus merge_procs snapshots
    $ ln -s snapshots/snapshots_s2.h5 restart.h5
    $ mpiexec -n 4 python3 rayleigh_benard.py

The simulations should take a few process-minutes to run.

"""

import numpy as np
from mpi4py import MPI
import time
import pathlib
import argparse

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(
        description='Simulation script for Rayleigh-Benard 2D using Dedalus')
    parser.add_argument('--lx', default=4.0, type=float,
                        help='Physical length in x dimension. (default: 4.0)')
    parser.add_argument('--lz', default=1.0, type=float,
                        help='Physical length in z dimension. (default: 1.0)')
    parser.add_argument('--res_x', default=512, type=int,
                        help='Simulation resolution in x dimension. (default: 512)')
    parser.add_argument('--res_z', default=128, type=int,
                        help='Simulation resolution in z dimension. (default: 128)')
    parser.add_argument('--dt', default=0.125, type=float,
                        help='Simulation step size in time. (default: 0.125)')
    parser.add_argument('--stop_sim_time', default=50., type=float,
                        help='Simulation stop time. (default: 50)')
    parser.add_argument('--rayleigh', default=1e6, type=float,
                        help='Simulation Rayleigh number. (default: 1e6)')
    parser.add_argument('--prandtl', default=1., type=float,
                        help='Simulation Prandtl number. (default: 1.0)')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed for initial perturbations. (default: 42)')

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    # Parameters
    Lx, Lz = (args.lx, args.lz)
    Prandtl = args.prandtl
    Rayleigh = args.rayleigh
    seed = args.seed

    # Create bases and domain
    x_basis = de.Fourier('x', args.res_x, interval=(0, Lx), dealias=3/2)  # 256
    z_basis = de.Chebyshev('z', args.res_z, interval=(-Lz/2, Lz/2), dealias=3/2)  # 64
    domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

    # 2D Boussinesq hydrodynamics
    problem = de.IVP(domain, variables=['p','b','u','w','bz','uz','wz'])
    problem.meta['p','b','u','w']['z']['dirichlet'] = True
    problem.parameters['P'] = (Rayleigh * Prandtl)**(-1/2)
    problem.parameters['R'] = (Rayleigh / Prandtl)**(-1/2)
    problem.parameters['F'] = F = 1
    problem.add_equation("dx(u) + wz = 0")
    # problem.add_equation("dt(b) - P*(dx(dx(b)) + dz(bz)) - F*w       = -(u*dx(b) + w*bz)")
    problem.add_equation("dt(b) - P*(dx(dx(b)) + dz(bz))             = -(u*dx(b) + w*bz)")
    problem.add_equation("dt(u) - R*(dx(dx(u)) + dz(uz)) + dx(p)     = -(u*dx(u) + w*uz)")
    problem.add_equation("dt(w) - R*(dx(dx(w)) + dz(wz)) + dz(p) - b = -(u*dx(w) + w*wz)")
    problem.add_equation("bz - dz(b) = 0")
    problem.add_equation("uz - dz(u) = 0")
    problem.add_equation("wz - dz(w) = 0")
    problem.add_bc("left(b) = 0.5")
    problem.add_bc("left(u) = 0")
    problem.add_bc("left(w) = 0")
    problem.add_bc("right(b) = -0.5")
    problem.add_bc("right(u) = 0")
    problem.add_bc("right(w) = 0", condition="(nx != 0)")
    problem.add_bc("right(p) = 0", condition="(nx == 0)")

    # Build solver
    solver = problem.build_solver(de.timesteppers.RK222)
    logger.info('Solver built')

    # Initial conditions or restart
    if not pathlib.Path('restart.h5').exists():

        # Initial conditions
        x = domain.grid(0)
        z = domain.grid(1)
        b = solver.state['b']
        bz = solver.state['bz']

        # Random perturbations, initialized globally for same results in parallel
        gshape = domain.dist.grid_layout.global_shape(scales=1)
        slices = domain.dist.grid_layout.slices(scales=1)
        rand = np.random.RandomState(seed=seed)
        noise = rand.standard_normal(gshape)[slices]

        # Linear background + perturbations damped at walls
        zb, zt = z_basis.interval
        pert =  1e-3 * noise * (zt - z) * (z - zb)
        b['g'] += F * pert
        b.differentiate('z', out=bz)

        # Timestepping and output
        dt = args.dt
        stop_sim_time = args.stop_sim_time
        fh_mode = 'overwrite'

    else:
        # Restart
        write, last_dt = solver.load_state('restart.h5', -1)

        # Timestepping and output
        dt = last_dt
        stop_sim_time = args.stop_sim_time
        fh_mode = 'append'

    # Integration parameters
    solver.stop_sim_time = stop_sim_time

    # Analysis
    snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.25, max_writes=50, mode=fh_mode)
    snapshots.add_system(solver.state)

    # CFL
    CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=1,
                         max_change=1.5, min_change=0.5, max_dt=0.125, threshold=0.05)
    CFL.add_velocities(('u', 'w'))

    # Flow properties
    flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
    flow.add_property("sqrt(u*u + w*w) / R", name='Re')

    # Main loop
    try:
        logger.info('Starting loop')
        start_time = time.time()
        while solver.proceed:
            dt = CFL.compute_dt()
            dt = solver.step(dt)
            if (solver.iteration-1) % 10 == 0:
                logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
                logger.info('Max Re = %f' %flow.max('Re'))
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        end_time = time.time()
        logger.info('Iterations: %i' %solver.iteration)
        logger.info('Sim end time: %f' %solver.sim_time)
        logger.info('Run time: %.2f sec' %(end_time-start_time))
        logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))

if __name__ == '__main__':
    main()