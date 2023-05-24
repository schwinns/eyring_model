# Script to run Eyring Model analyses

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from time import perf_counter as timer

from eyring_model import EyringModel, Path

# Define global constants
global kB 
kB = 1.380649 * 10**-23    # Boltzmann (m^2 kg / s^2 K)
global h
h = 6.62607 * 10**-34      # Planck (m^2 kg / s)
global R
R = 1.9858775 * 10**-3     # universal gas (kcal / mol K)

def path_convergence(realizations, dH_barrier, dS_barrier, dH_sigma, dS_sigma, dist, T=300, multi=True):

    if dist in ['normal', 'N', 'norm']:
        params = {'mu'  : np.array([dH_barrier, dS_barrier]),
                  'cov' : np.array([[dH_sigma**2,0],
                                    [0,dS_sigma**2]])}
    
    elif dist in ['exponential', 'exp']:
        params = {'beta' : np.array([dH_barrier, dS_barrier])}
    
    elif dist in ['equal', 'single', 'none', None]:
        params = {'mu' : np.array([dH_barrier, dS_barrier])}

    # run this on all available processors

    def calculate_barriers_permeability(N, dist, dist_params, T=T, multi=multi):

        model = EyringModel(T=T)
        for n in range(N):
            model.add_Path(n_jumps=50, lam=10) # 10 Angstrom jump lengths over 50
            model.paths[n].generate_membrane_barriers(dist=dist, dist_params=dist_params, multi=multi)

        return model.calculate_effective_barrier(), model.calculate_permeability()

    ########### DISTRIBUTE THE WORK ACROSS PROCESSORS ##########

    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    rank = comm.Get_rank()
    
    my_start = timer()

    if rank == 0:
        print(f'\n{nprocs} processors calculating the path convergence of {dist} distributions of barriers')

        fig1, ax1 = plt.subplots(1,1)
        fig2, ax2 = plt.subplots(1,1)
        
        # Increasing number of paths to see how effective barrier changes (when it stabilizes)
        n_paths = np.array([50,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,
                        1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,
                        3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000,4100,4200,4300,
                        4400,4500,4600,4700,4800,4900,5000,5500,6000,6500,7000,
                        7500,8000,8500,9000,9500,10_000], dtype='i')
        
        # split up n_paths for each proc
        ave, res = divmod(n_paths.size, nprocs)
        count = np.array([ave + 1 if p < res else ave for p in range(nprocs)], dtype='i')

        # save the indices where each processor starts
        displ = np.array([sum(count[:p]) for p in range(nprocs)])

        # save to calculate average across realizations
        avg_barrier = np.zeros(len(n_paths))
        avg_permeability = np.zeros(len(n_paths))

    else:
        # initialize n_paths, count, displ on workers
        n_paths = None
        count = np.zeros(nprocs, dtype='i')
        displ = None

    # send count to all processors
    comm.Bcast(count, root=0)

    # initialize my_paths on all processors
    my_paths = np.zeros(count[rank], dtype='i')

    # break the array up and send to processors
    comm.Scatterv([n_paths, count, displ, MPI.INT], my_paths, root=0)

    # calculate effective barriers and permeabilities on each processor
    effective_barriers = np.zeros(len(my_paths))
    permeabilities = np.zeros(len(my_paths))

    ########## RUN A BUNCH OF REALIZATIONS ##########

    for r in range(realizations):

        if rank == 0:
            print(f'Realization {r+1}...')

        for i,N in enumerate(my_paths):
            effective_barriers[i], permeabilities[i] = calculate_barriers_permeability(N, dist, params)

        # create big arrays for Gather
        all_barriers = np.zeros(sum(count))
        all_permeabilities = np.zeros(sum(count))

        # gather all the barriers and permeabilities back to master for plotting
        comm.Gatherv(effective_barriers, [all_barriers, count, displ, MPI.DOUBLE], root=0)
        comm.Gatherv(permeabilities, [all_permeabilities, count, displ, MPI.DOUBLE], root=0)

        if rank == 0:
            # save for average across realizations 
            avg_barrier += all_barriers + R*T*np.log(n_paths)
            avg_permeability += all_permeabilities / n_paths

            ax1.plot(n_paths, all_barriers + R*T*np.log(n_paths), alpha=0.1, c='k')
            ax2.plot(n_paths, all_permeabilities / n_paths, alpha=0.1, c='k')
        

    if rank == 0:
        data = np.column_stack((n_paths, avg_barrier/realizations, avg_permeability/realizations))
        np.savetxt(f'avg_convergence_{dist}_{realizations}iter.csv', data, delimiter=',')
        ax1.plot(n_paths, avg_barrier / realizations, c='r')
        ax2.plot(n_paths, avg_permeability / realizations, c='r')

        ax1.set_xlabel('number of paths')
        ax1.set_ylabel('effective barrier')
        ax2.set_xlabel('number of paths')
        ax2.set_ylabel('permeability per path')
        plt.show()

    my_end = timer()
    print(f'Total time for rank {rank}: {my_end - my_start:.2f} s')


if __name__ == "__main__":

    # Inputs
    iters = 300
    T = 300
    multi = True
    dH_barrier = 4.5
    dS_barrier = -6/300
    dH_sigma = 1.5
    dS_sigma = 2/300

    dG_barrier = dH_barrier - T*dS_barrier

    # normal distributions
    path_convergence(iters, dH_barrier, dS_barrier, dH_sigma, dS_sigma, dist='norm', T=T, multi=multi)

    # exponential distributions
    path_convergence(iters, dH_barrier, dS_barrier, dH_sigma, dS_sigma, dist='exp', T=T, multi=multi)
