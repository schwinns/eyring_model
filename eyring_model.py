# EyringModel class

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from exceptions import *

# Define global constants
global kB 
kB = 1.380649 * 10**-23    # Boltzmann (m^2 kg / s^2 K)
global h
h = 6.62607 * 10**-34      # Planck (m^2 kg / s)
global R
R = 1.9858775 * 10**-3     # universal gas (kcal / mol K)


class EyringModel():
    def __init__(self, T=300, n_jumps=200, lam=2, barrier_sm=5, barrier_ms=5):
        ''' Eyring model for transport through a membrane as a series of molecular jumps (see original citation: https://doi.org/10.1021/j150474a012)

        :param T: temperature (K), default=300
        :param n_jumps: number of membrane jumps in direction of transport, default=200
        :param lam: average membrane jump length (Angstroms), default=2
        :param barrier_sm: barrier for the solution-membrane interface, default=5 (becomes 5*R*T)
        :param barrier_ms: barrier for the membrane-solution interface, default=5 (becomes 5*R*T)

        :type T: float
        :type n_jumps: int
        :type lam: float
        :type barrier_sm: float
        :type barrier_ms: float

        '''

        # Initialize based on inputs
        self.T = T
        self.n_jumps = n_jumps
        self.lam = lam

        # Set some initial values
        self.seed = None
        self.membrane_barriers = None

        # Default values for some parameters
        self.generate_jump_distribution(dist_params={'mu' : lam}) # jump length distributions
        self.lam_sm = lam                                # solution-membrane jump length
        self.lam_ms = lam                                # membrane-solution jump length
        
        self.barrier_sm = barrier_sm*R*T             # solution-membrane barrier
        self.barrier_ms = barrier_ms*R*T             # membrane-solution barrier


    def generate_membrane_barriers(self, dist='normal', dist_params={'mu': 0, 'sigma' : 1}, seed=None):
        '''Generate a distribution of membrane barriers
        
        :param dist: distribution from which to draw the membrane barriers, default='normal'
        :param dist_params: parameters for the membrane barrier distribution, default={'mu' : 0, 'sigma': 1}
        :param seed: seed for the random number generator, default=None

        :type dist: str
        :type dist_params: dict
        :type seed: int

        '''

        # dictionary error
        if not isinstance(dist_params, dict):
                raise DistributionError('Provide distribution parameters as a dictionary')

        # create random number generator   
        rng = np.random.default_rng(seed)

        if dist in ['normal', 'N', 'norm']: # normal distribution of barriers
            # Raise an error if the correct parameters are not provided
            if 'mu' not in dist_params.keys():
                raise DistributionError("'mu' must be a key in distribution parameters for a normal distribution")
            elif 'sigma' not in dist_params.keys():
                raise DistributionError("'sigma' must be a key in distribution parameters for a normal distribution")
            
            # generate barrier distribution
            self.membrane_barriers = rng.normal(loc=dist_params['mu'], scale=dist_params['sigma'], size=self.n_jumps)

        elif dist in ['exponential', 'exp']: # exponential distribution of barriers
            # Raise an error if the correct parameters are not provided
            if 'beta' not in dist_params.keys():
                raise DistributionError("'beta' must be a key in distribution parameters for an exponential distribution")

            # generate barrier distribution
            self.membrane_barriers = rng.exponential(scale=dist_params['beta'], size=self.n_jumps)

        elif dist in ['uniform', 'uni']: # uniform distribution of barriers
            # Raise an error if the correct parameters are not provided
            if 'a' not in dist_params.keys():
                raise DistributionError("'a' must be a key in distribution parameters for a normal distribution")
            elif 'b' not in dist_params.keys():
                raise DistributionError("'b' must be a key in distribution parameters for a normal distribution")
            
            # generate barrier distribution
            self.membrane_barriers = rng.uniform(low=dist_params['a'], high=dist_params['b'], size=self.n_jumps)

        elif dist in ['equal', 'single', 'none', None]: # no distribution of barriers -- assumes single barrier
            # Raise an error if the correct parameters are not provided
            if 'mu' not in dist_params.keys():
                raise DistributionError("'mu' must be a key in distribution parameters for no distribution of barriers")

            # generate barrier distribution
            self.membrane_barriers = np.ones(self.n_jumps) * dist_params['mu']

        else: # other distributions not implemented
            # Raise an error if other distribution is provided
            dist_options = ['normal', 'exponential', 'uniform', 'equal']
            raise DistributionError('{} is not currently implemented. Try one of {}'.format(dist, dist_options))

        # print('Generated {} distribution of membrane barriers with mean {:.4f} and stdev {:.4f}'.format(dist, self.membrane_barriers.mean(), self.membrane_barriers.std()))
        return self.membrane_barriers

    
    def generate_jump_distribution(self, dist=None, dist_params={'mu' : 2}, seed=None):
        '''Generate a distribution of jump lengths

        :param dist: distribution from which to draw the jump lengths, default=None (which means all jump lengths are the same)
        :param dist_params: parameters for the membrane barrier distribution, default={'mu' : 2}
        :param seed: seed for the random number generator

        :type dist: str
        :type dist_params: dict
        :type seed: int

        '''

        # dictionary error
        if not isinstance(dist_params, dict):
                raise DistributionError('Provide distribution parameters as a dictionary')

        # create random number generator   
        rng = np.random.default_rng(seed)

        if dist in ['equal', 'single', 'none', None]: # all the same length
            # Raise an error if the correct parameters are not provided
            if 'mu' not in dist_params.keys():
                raise DistributionError("'mu' must be a key in distribution parameters for a normal distribution")

            self.jump_lengths = np.ones(self.n_jumps) * dist_params['mu']

        elif dist in ['normal', 'N', 'norm']: # normal distribution of barriers
            # Raise an error if the correct parameters are not provided
            if 'mu' not in dist_params.keys():
                raise DistributionError("'mu' must be a key in distribution parameters for a normal distribution")
            elif 'sigma' not in dist_params.keys():
                raise DistributionError("'sigma' must be a key in distribution parameters for a normal distribution")
            
            # generate barrier distribution
            self.jump_lengths = rng.normal(loc=dist_params['mu'], scale=dist_params['sigma'], size=self.n_jumps)

        elif dist in ['exponential', 'exp']: # exponential distribution of barriers
            # Raise an error if the correct parameters are not provided
            if 'beta' not in dist_params.keys():
                raise DistributionError("'beta' must be a key in distribution parameters for an exponential distribution")

            # generate barrier distribution
            self.jump_lengths = rng.exponential(scale=dist_params['beta'], size=self.n_jumps)

        elif dist in ['uniform', 'uni']: # uniform distribution of barriers
            # Raise an error if the correct parameters are not provided
            if 'a' not in dist_params.keys():
                raise DistributionError("'a' must be a key in distribution parameters for a normal distribution")
            elif 'b' not in dist_params.keys():
                raise DistributionError("'b' must be a key in distribution parameters for a normal distribution")
            
            # generate barrier distribution
            self.jump_lengths = rng.uniform(low=dist_params['a'], high=dist_params['b'], size=self.n_jumps)

        else: # other distributions not implemented
            # Raise an error if other distribution is provided
            dist_options = ['normal', 'exponential', 'uniform', 'equal']
            raise DistributionError('{} is not currently implemented. Try one of {}'.format(dist, dist_options))        


        # print('Generated {} distribution of jump lengths with mean {:.4f} and stdev {:.4f}'.format(dist, self.jump_lengths.mean(), self.jump_lengths.std()))


    def _P_membrane_barriers(self, T=None):
        '''Calculate membrane transport contribution to permeability from distribution of jump lengths and barriers
        
        :param T: temperature (K), default=None (use self.T)

        :type T: float

        '''

        if self.membrane_barriers is None:
            raise DistributionError('No membrane barrier distribution. Use generate_membrane_barriers method')

        if T is None:
            temp = self.T
        else:
            temp = T

        A = h / (kB*temp) * 60 * 60
        exp = np.exp(- self.membrane_barriers / (R*temp))
        lam = self.jump_lengths / 10**10
        S = np.sum( 1 / (self.jump_lengths * exp) )

        return A*S # units = h / m


    def _P_interfacial_barriers(self, T=None):
        '''Calculate membrane partitioning contribution to permeability from interfacial barriers
        
        :param T: temperature (K), default=None (use self.T)

        :type T: float

        '''

        if self.membrane_barriers is None:
            raise DistributionError('No membrane barrier distribution. Use generate_membrane_barriers method')

        if T is None:
            temp = self.T
        else:
            temp = T

        A = self.lam_sm/self.lam_ms
        exp = np.exp((self.barrier_ms - self.barrier_sm) / R / temp)

        return A * exp # unitless

    
    def calculate_permeability(self, T=None):
        '''Calculate permeabilitty from membrane barriers
        
        :param T: temperature (K), default=None (use self.T)

        :type T: float
        
        '''
        num  = self._P_interfacial_barriers(T=T)
        den = self._P_membrane_barriers(T=T)
        return num / den * 1000 # units = L / m^2 / h
    

    def calculate_effective_barrier_single_path(self, T=None):
        '''Calculate the effective barrier that is experimentally observed from membrane barriers
        
        :param T: temperature (K), default=None (use self.T)

        :type T: float
        
        '''

        if T is None:
            temp = self.T
        else:
            temp = T

        delta = self.jump_lengths.sum()
        num = (self.lam_sm / self.lam_ms) * np.exp((self.barrier_ms - self.barrier_sm) / (R*temp))
        den = (self.lam**2 / delta) * np.sum( np.exp(self.membrane_barriers / (R*temp)) / self.jump_lengths)
        return -np.log(num / den) * R*temp


    def calculate_dH_dS(self, temps, mem_dist='normal', mem_dist_params={'mu' : 0, 'sigma' : 1}, jump_dist=None, jump_dist_params={'mu' : 2}, plot=False):
        '''Calculate the enthalpy and entropy from permeabilities measured over temperature
        
        :param temps: temperatures over which to estimate dH and dS (K)
        :param mem_dist: distribution from which to draw the membrane barriers, default='normal'
        :param mem_dist_params: parameters for the membrane barrier distribution, default={'mu' : 0, 'sigma': 1}
        :param jump_dist: distribution from which to draw the jump lengths, default=None (which means all jump lengths are the same)
        :param jump_dist_params: parameters for the membrane barrier distribution, default={'mu' : 2}
        :param plot: plot the linear fit used to determine dH and dS, default=False

        :type temps: float, array-like
        :type mem_dist: str
        :type mem_dist_params: dict
        :type jump_dist: str
        :type jump_dist_params: dict
        :type plot: bool

        '''
        
        X = np.zeros(len(temps))
        Y = np.zeros(len(temps))
        i = 0
        for T in temps:
            self.barrier_ms = R * T
            self.barrier_sm = R * T
            self.generate_membrane_barriers(mem_dist, mem_dist_params)
            self.generate_jump_distribution(jump_dist, jump_dist_params)

            P = self.calculate_permeability(T=T)
            delta = np.sum(self.jump_lengths)
            Y[i] = np.log(P*h*delta / (kB*T*self.lam**2))
            X[i] = 1 / T        
            i += 1
        
        data = pd.DataFrame(np.array([X,Y]).T, columns=['1/T', 'ln(P h del / kB T lam^2)'])
        m, b = np.polyfit(X,Y,1)
        if plot:
            x = np.linspace(X[0], X[-1], num=50)
            sns.lmplot(x='1/T', y='ln(P h del / kB T lam^2)', data=data, scatter_kws={'alpha':0.75, 'edgecolor':'tab:blue'})
            plt.xlabel('1/T')
            plt.ylabel('ln($P h \delta$ / $k_B T \lambda$)')
            xmin, xmax = plt.xlim()
            ymin, ymax = plt.ylim()
            plt.text(xmax*0.95, ymax*1.05, 'dH = {:.4f}\ndS = {:.4f}'.format(-m*R, b*R), ha='right')
            plt.show()
            
        dH = -m*R
        dS = b*R
        return dH, dS


    def plot_distribution(self, bw=None, show_hist=False, n_bins=50, label=None, ax=None):

        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=(10,6))

        if show_hist:
            sns.histplot(self.membrane_barriers, bins=n_bins, 
                         stat='density')
            sns.kdeplot(self.membrane_barriers, bw_method=bw, c='r', label=label)
        else:
            sns.kdeplot(self.membrane_barriers, bw_method=bw, label=label, fill=True)
        
        plt.xlabel('$\Delta G_{M,j}$')

        return ax



if __name__ == '__main__':

    # Choose what analyses to run
    parallel_pores = True
    surf3d = False
    compare_effective_barriers = False
    estimate_dH_dS = False
    tabulated_comparisons = False


    # Inputs for testing barriers
    T = 300
    large_barrier = 18*R*T
    small_barrier = large_barrier / 2
    sigma = large_barrier / 3

    if parallel_pores:

        from tqdm import tqdm

        n_pores = 50

        model = EyringModel(T=T, barrier_ms=10, barrier_sm=10)
        params = {'mu' : large_barrier, 'sigma' : sigma}

        fig, ax = plt.subplots(2,1, figsize=(12,8), sharex=True)

        # NORMAL DISTRIBUTION OF BARRIERS

        # loop through each pore and calculate individual permeabilities/effective barriers
        # plot the membrane barrier distribution for each pore, overlapping
        permeabilities = np.zeros(n_pores)
        mem_dist = np.zeros((model.n_jumps, n_pores))
        eff_barriers = np.zeros(n_pores)
        for n in tqdm(range(n_pores)):
            mem_dist[:,n] = model.generate_membrane_barriers(dist='normal', dist_params=params) / (R*T)
            permeabilities[n] = model.calculate_permeability()
            eff_barriers[n] = model.calculate_effective_barrier_single_path() / (R*T)
            sns.kdeplot(mem_dist[:,n], bw_method='scott', fill=False, ax=ax[0])

        # save data as pandas DataFrame
        df = pd.DataFrame()
        df['pores'] = np.arange(1,n_pores+1)
        df['permeability'] = permeabilities
        df['effective_barriers'] = eff_barriers
        df['permeability_percent'] = permeabilities / permeabilities.sum() * 100
        
        # Calculate the effective barrier for the parallel pores, i.e. overall permeability
        # plot the effective barrier, max barrier, and mean barrier
        delta = model.jump_lengths.sum()
        effective_barrier = -np.log(delta / model.lam**2 * h / kB / T * permeabilities.sum())
        ax[0].axvline(effective_barrier, ls='dashed', c='k', label='effective barrier')
        ax[0].axvline(mem_dist.max(), ls='dashed', c='b', label='maximum individual barrier')
        ax[0].axvline(mem_dist.mean(), ls='dashed', c='r', label='mean barrier')
        ax[0].legend()
        ax[0].set_title(f'Normal distribution, mean = {large_barrier/R/T:.0f}RT, stdev = {sigma/R/T:.0f}RT')

        fig1, ax1 = plt.subplots(1,1, figsize=(6,6))
        sns.barplot(data=df, x='pores', y='permeability_percent', ax=ax1)
        ax1.set_ylabel('percentage of permeability')
        # plt.axhline(permeabilities.sum(), c='k', ls='dashed')
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        ax1.text(xmax*0.95, ymax*0.9, 'Max P: {:.4f}\nOverall P: {:.4f}'.format(permeabilities.max(), permeabilities.sum()), ha='right')

        # EXPONENTIAL DISTRIBUTION OF BARRIERS

        params = {'beta' : large_barrier}

        # loop through each pore and calculate individual permeabilities/effective barriers
        # plot the membrane barrier distribution for each pore, overlapping
        permeabilities = np.zeros(n_pores)
        mem_dist = np.zeros((model.n_jumps, n_pores))
        for n in tqdm(range(n_pores)):
            mem_dist[:,n] = model.generate_membrane_barriers(dist='exponential', dist_params=params) / (R*T)
            permeabilities[n] = model.calculate_permeability()
            eff_barriers[n] = model.calculate_effective_barrier_single_path() / (R*T)
            sns.kdeplot(mem_dist[:,n], bw_method='scott', fill=False, ax=ax[1])
        
        # save data as pandas DataFrame
        df = pd.DataFrame()
        df['pores'] = np.arange(1,n_pores+1)
        df['permeability'] = permeabilities
        df['effective_barriers'] = eff_barriers
        df['permeability_percent'] = permeabilities / permeabilities.sum() * 100
        
        # Calculate the effective barrier for the parallel pores, i.e. overall permeability
        # plot the effective barrier, max barrier, and mean barrier
        delta = model.jump_lengths.sum()
        effective_barrier = -np.log(delta / model.lam**2 * h / kB / T * permeabilities.sum())
        ax[1].axvline(effective_barrier, ls='dashed', c='k', label='effective barrier')
        ax[1].axvline(mem_dist.max(), ls='dashed', c='b', label='maximum individual barrier')
        ax[1].axvline(mem_dist.mean(), ls='dashed', c='r', label='mean barrier')
        ax[1].set_title(f'Exponential distribution, mean = {large_barrier/R/T:.0f}RT')

        ax[1].set_xlabel('$\Delta G_{M,j} / RT$')
        ax[1].legend()
        # fig.suptitle('Distributions of membrane barriers for 50 parallel pores')

        fig2, ax2 = plt.subplots(1,1, figsize=(6,6))
        sns.barplot(data=df, x='pores', y='permeability_percent', ax=ax2)
        ax2.set_ylabel('percentage of permeability')
        # plt.axhline(permeabilities.sum(), c='k', ls='dashed')
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        ax2.text(xmax*0.95, ymax*0.9, 'Max P: {:.4f}\nOverall P: {:.4f}'.format(permeabilities.max(), permeabilities.sum()), ha='right')
        plt.show()


    if surf3d:

        from matplotlib import cm

        n_pores = 50

        model = EyringModel(T=T, barrier_ms=10, barrier_sm=10)
        params = {'mu' : large_barrier, 'sigma' : sigma}

        for n in range(n_pores):
            mem_dist[:,n] = model.generate_membrane_barriers(dist='normal', dist_params=params) / (R*T)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X = np.arange(n_pores)
        Y = np.arange(model.n_jumps)
        X, Y = np.meshgrid(X, Y)
        Z = mem_dist
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0)
        fig.colorbar(surf, label='barriers')
        plt.xlabel('pores')
        plt.ylabel('thickness')
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        yticks = np.arange(n_pores)
        for k in yticks:
            xs = np.arange(model.n_jumps)
            ys = mem_dist[:,k]
            ax.bar(xs, ys, zs=k, zdir='y', alpha=0.9)

        ax.set_xlabel('thickness')
        ax.set_ylabel('pores')
        ax.set_zlabel('barriers')
        ax.set_yticks(yticks)
        plt.show()
        

    if compare_effective_barriers:

        model = EyringModel(T=T, barrier_ms=10, barrier_sm=10)
        params = {'mu' : large_barrier, 'sigma' : sigma}
        model.generate_membrane_barriers(dist='normal', dist_params=params)
        dG_eff = model.calculate_effective_barrier_single_path() / (R*T)
        model.membrane_barriers = model.membrane_barriers / (R*T)
        ax = model.plot_distribution(bw='scott', show_hist=False, label='N(30RT, 10RT)')
        ax.axvline(dG_eff, ls='dashed', c='k', label='N: $\Delta G_{eff}$/RT = %.4f' % (dG_eff))

        params = {'beta' : large_barrier}
        model.generate_membrane_barriers(dist='exponential', dist_params=params)
        dG_eff = model.calculate_effective_barrier_single_path() / (R*T)
        model.membrane_barriers = model.membrane_barriers / (R*T)
        model.plot_distribution(bw='scott', show_hist=False, ax=ax, label='exp(30RT)')
        ax.axvline(dG_eff, ls='dotted', c='k', label='exp: $\Delta G_{eff}$/RT = %.4f' % (dG_eff))
        
        ax.axvline(large_barrier/R/T, c='r', label='mean')
        ax.set_xlabel('$\Delta G_{M,j}$ / RT')
        plt.legend(loc='upper center')
        plt.savefig('effective_barrier_distribution_comparison.png')
        plt.show()
    
    if estimate_dH_dS:

        model = EyringModel(T=T, barrier_ms=10, barrier_sm=10)
        params = {'mu' : large_barrier, 'sigma' : sigma}
        temps = [250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350]*10
        dH, dS = model.calculate_dH_dS(temps, mem_dist='normal', mem_dist_params=params, plot=True)

    if tabulated_comparisons:

        from tabulate import tabulate

        table = []
        for barrier in [small_barrier, large_barrier]:
            
            dist_params = [
                {'mu' : barrier}, # equal barriers with mean = barrier
                {'mu' : barrier, 'sigma' : sigma}, # normal distribution with mean = barrier
                {'beta' : barrier}, # exponential distribution with mean = barrier
                {'a' : 1/2 * barrier, 'b' : 3/2 * barrier} # gives uniform distribution with mean = barrier
            ]
            
            for d, dist in enumerate(['equal', 'normal', 'exponential', 'uniform']):
                
                mem_params = dist_params[d]
                model = EyringModel(T=T)
                model.generate_membrane_barriers(dist, mem_params)
                P = model.calculate_permeability(T=T)
                row = [dist, '{}RT'.format(int(barrier / R / T)), P]
                table.append(row)


        print()
        print(tabulate(table, headers=['Membrane distribution', 'Average barrier', 'Permeability (L/m^2/h)']))


        # Jump length distributions to test
        large_jump = 5
        small_jump = 0.5
        sigma = 0.5
        mem_params = {'mu' : 15*R*T, 'sigma' : R*T}

        table = []
        for jump in [small_jump, large_jump]:
            
            dist_params = [
                {'mu' : jump}, # equal jumps with mean = jump
                {'mu' : jump, 'sigma' : sigma}, # normal distribution with mean = jump
                {'beta' : jump}, # exponential distribution with mean = jump
                {'a' : 1/2 * jump, 'b' : 3/2 * jump} # gives uniform distribution with mean = jump
                ]
            
            for d, dist in enumerate(['equal', 'normal', 'exponential', 'uniform']):
                
                jump_params = dist_params[d]
                model = EyringModel(T=T)
                model.generate_jump_distribution(dist, dist_params[d])
                model.generate_membrane_barriers(dist_params=mem_params)
                P = model.calculate_permeability(T=T)
                row = [dist, jump, P]
                table.append(row)


        print()
        print(tabulate(table, headers=['Jump distribution', 'Average jump', 'Permeability (L/m^2/h)']))
