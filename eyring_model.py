# EyringModel class

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from exceptions import *

# Define global constants
global kB 
kB = 1.380649 * 10**-23    # Boltzmann
global h
h = 6.62607 * 10**-34      # Planck
global R
R = 1.9858775 * 10**-3     # universal gas (kcal / mol / K)


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


    def _P_mem_bar(self, T=None):
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

        return np.sum( np.exp(self.membrane_barriers / R / temp) / self.jump_lengths ) # units 1/length


    def _P_inter_bar(self, T=None):
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

        return (self.lam_sm/self.lam_ms) * (kB*temp/h) * np.exp((self.barrier_ms - self.barrier_sm) / R / temp) # units 1/s

    
    def calculate_permeability(self, T=None):
        '''Calculate permeabilitty from membrane barriers
        
        :param T: temperature (K), default=None (use self.T)

        :type T: float
        
        '''
        num  = self._P_inter_bar(T=T)
        den = self._P_mem_bar(T=T)
        return num / den / 10 / (10**9) * 1000 * 60 * 60
    

    def calculate_effective_barrier(self, T=None):
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
    compare_effective_barriers = False
    estimate_dH_dS = False
    tabulated_comparisons = False


    # Inputs for testing barriers
    T = 300
    large_barrier = 30*R*T
    small_barrier = 15*R*T
    sigma = 10*R*T

    if parallel_pores:

        from tqdm import tqdm

        n_pores = 10**6

        model = EyringModel(T=T, barrier_ms=10, barrier_sm=10)
        params = {'mu' : large_barrier, 'sigma' : sigma}

        permeabilities = np.zeros(n_pores)
        for n in tqdm(range(n_pores)):
            model.generate_membrane_barriers(dist='normal', dist_params=params)
            permeabilities[n] = model.calculate_permeability()

        # sns.histplot(permeabilities, bins=100, stat='density')
        sns.kdeplot(permeabilities, bw_method='scott', fill=True)
        print(permeabilities.sum())
        plt.show()

    if compare_effective_barriers:

        model = EyringModel(T=T, barrier_ms=10, barrier_sm=10)
        params = {'mu' : large_barrier, 'sigma' : sigma}
        model.generate_membrane_barriers(dist='normal', dist_params=params)
        dG_eff = model.calculate_effective_barrier() / (R*T)
        model.membrane_barriers = model.membrane_barriers / (R*T)
        ax = model.plot_distribution(bw='scott', show_hist=False, label='N(30RT, 10RT)')
        ax.axvline(dG_eff, ls='dashed', c='k', label='N: $\Delta G_{eff}$/RT = %.4f' % (dG_eff))

        params = {'beta' : large_barrier}
        model.generate_membrane_barriers(dist='exponential', dist_params=params)
        dG_eff = model.calculate_effective_barrier() / (R*T)
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
