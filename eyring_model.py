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


class EyringModel:
    def __init__(self, T=300):
        ''' Eyring model for transport through a membrane assuming many parallel paths with separate permeabilities P_i
        
        :param T: temperature (K), default=300
        
        :type T: float
        
        '''

        self.paths = []
        self.T = T
        self.permeabilities = None
        self.deltas = []


    def add_Path(self, dist='normal', dist_params={'mu': 0, 'sigma' : 1}, n_jumps=200, lam=2, barrier_sm=5, barrier_ms=5):
        ''' Add path to the Eyring model with a given distribution of membrane barriers
        
        :param dist: distribution from which to draw the membrane barriers, default='normal'
        :param dist_params: parameters for the membrane barrier distribution, default={'mu' : 0, 'sigma': 1}
        :param n_jumps: number of membrane jumps in direction of transport, default=200
        :param lam: average membrane jump length (Angstroms), default=2
        :param barrier_sm: barrier for the solution-membrane interface, default=5 (becomes 5*R*T)
        :param barrier_ms: barrier for the membrane-solution interface, default=5 (becomes 5*R*T)

        :type dist: str
        :type dist_params: dict
        :type n_jumps: int
        :type lam: float
        :type barrier_sm: float
        :type barrier_ms: float

        '''
    
        p = Path(T=self.T, dist=dist, dist_params=dist_params, # membrane barriers 
                 n_jumps=n_jumps, lam=lam, # jump length distribution
                 barrier_sm=barrier_sm*R*T, barrier_ms=barrier_ms*R*T) # interfacial barriers
        
        self.paths.append(p)
        self.deltas.append(p.jump_lengths.sum())
        self.n_paths = len(self.paths)


    def get_lambda(self):
        ''' Get average jump length from the single paths'''

        return np.array([p.jump_lengths.mean() for p in self.paths]).mean()

    
    def calculate_permeability(self):
        ''' Calculate overall permeability as a sum of single path permeabilities'''

        self.permeabilities = np.zeros(self.n_paths)
        for n,p in enumerate(self.paths):
            self.permeabilities[n] = p.calculate_permeability()

        self.permeability = self.permeabilities.sum()
        return self.permeability
    

    def calculate_effective_barrier(self):
        ''' Calculate overall effective barrier from parallel permeabilities'''

        if self.permeabilities is None:
            P = self.calculate_permeability()

        lam = self.get_lambda()
        delta = np.array(self.deltas).mean()

        self.effective_barrier = -np.log(delta / lam**2 * h / kB / self.T * self.permeabilities.sum())

        return self.effective_barrier


class Path:
    def __init__(self, T=300, dist='normal', dist_params={'mu': 0, 'sigma' : 1}, n_jumps=200, lam=2, barrier_sm=5, barrier_ms=5):
        ''' Single path of the Eyring model for transport through a membrane as a series of molecular jumps (see original citation: https://doi.org/10.1021/j150474a012)

        :param T: temperature (K), default=300
        :param dist: distribution from which to draw the membrane barriers, default='normal'
        :param dist_params: parameters for the membrane barrier distribution, default={'mu' : 0, 'sigma': 1}
        :param n_jumps: number of membrane jumps in direction of transport, default=200
        :param lam: average membrane jump length (Angstroms), default=2
        :param barrier_sm: barrier for the solution-membrane interface, default=5
        :param barrier_ms: barrier for the membrane-solution interface, default=5

        :type T: float
        :type dist: str
        :type dist_params: dict
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
        self.generate_membrane_barriers(dist=dist, dist_params=dist_params)
        self.generate_jump_distribution(dist_params={'mu' : lam}) # jump length distributions
        self.lam_sm = lam                                # solution-membrane jump length
        self.lam_ms = lam                                # membrane-solution jump length
        
        self.barrier_sm = barrier_sm             # solution-membrane barrier
        self.barrier_ms = barrier_ms             # membrane-solution barrier


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
        self.permeability = num / den * 1000
        return num / den * 1000 # units = L / m^2 / h
    

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


    def plot_distribution(self, bw=None, show_hist=False, n_bins=50, label=None, fill=True, ax=None):

        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=(10,6))

        if show_hist:
            sns.histplot(self.membrane_barriers, bins=n_bins, 
                         stat='density', ax=ax)
            sns.kdeplot(self.membrane_barriers, bw_method=bw, c='r', label=label, ax=ax)
        else:
            sns.kdeplot(self.membrane_barriers, bw_method=bw, label=label, fill=fill, ax=ax)
        
        plt.xlabel('$\Delta G_{M,j}$')

        return ax


if __name__ == '__main__':

    # Choose what analyses to run
    parallel_pores = True
    compare_effective_barriers = True
    estimate_dH_dS = True

    # Inputs for testing barriers
    T = 300
    large_barrier = 15*R*T
    small_barrier = large_barrier / 2
    sigma = large_barrier / 3

    if parallel_pores:

        from tqdm import tqdm

        n_paths = 50

        fig, ax = plt.subplots(2,1, figsize=(12,8), sharex=True)

        # NORMAL DISTRIBUTION OF BARRIERS

        model_norm = EyringModel(T=T)
        dist = 'normal'
        params = {'mu' : large_barrier, 'sigma' : sigma}

        # plot the membrane barrier distribution for each pore, overlapping
        effective_barriers = np.zeros(n_paths)
        for n in range(n_paths):
            model_norm.add_Path(dist=dist, dist_params=params)
            effective_barriers[n] = model_norm.paths[n].calculate_effective_barrier() / (R*T)
            model_norm.paths[n].plot_distribution(fill=False, ax=ax[0])

        permeability = model_norm.calculate_permeability()
        effective_barrier_norm = model_norm.calculate_effective_barrier()

        # save data as pandas DataFrame
        df_norm = pd.DataFrame()
        df_norm['pores'] = np.arange(1,n_paths+1)
        df_norm['permeability'] = model_norm.permeabilities
        df_norm['effective_barriers'] = effective_barriers
        df_norm['permeability_percent'] = model_norm.permeabilities / model_norm.permeabilities.sum() * 100
        
        # EXPONENTIAL DISTRIBUTION OF BARRIERS

        model_exp = EyringModel(T=T)
        dist = 'exponential'
        params = {'beta' : large_barrier}

        # plot the membrane barrier distribution for each pore, overlapping
        effective_barriers = np.zeros(n_paths)
        for n in range(n_paths):
            model_exp.add_Path(dist=dist, dist_params=params)
            effective_barriers[n] = model_exp.paths[n].calculate_effective_barrier() / (R*T)
            model_exp.paths[n].plot_distribution(fill=False, ax=ax[1])

        permeability = model_exp.calculate_permeability()
        effective_barrier_exp = model_exp.calculate_effective_barrier()

        # save data as pandas DataFrame
        df_exp = pd.DataFrame()
        df_exp['pores'] = np.arange(1,n_paths+1)
        df_exp['permeability'] = model_exp.permeabilities
        df_exp['effective_barriers'] = effective_barriers
        df_exp['permeability_percent'] = model_exp.permeabilities / model_exp.permeabilities.sum() * 100

        # PLOTTING

        # plot the effective barrier, max barrier, and mean barrier
        ax[0].axvline(effective_barrier_norm, ls='dashed', c='k', label='effective barrier')
        ax[0].axvline(large_barrier/R/T, ls='dashed', c='r', label='mean barrier')
        ax[0].legend()
        ax[0].set_title(f'Normal distribution, mean = {large_barrier/R/T:.0f}RT, stdev = {sigma/R/T:.0f}RT')

        ax[1].axvline(effective_barrier_exp, ls='dashed', c='k', label='effective barrier')
        ax[1].axvline(large_barrier/R/T, ls='dashed', c='r', label='mean barrier')
        ax[1].legend()
        ax[1].set_title(f'Exponential distribution, mean = {large_barrier/R/T:.0f}RT')

        ax[1].set_xlabel('$\Delta G_{M,j} / RT$')

        fig1, ax1 = plt.subplots(1,1, figsize=(6,6))
        sns.barplot(data=df_norm, x='pores', y='permeability_percent', ax=ax1)
        ax1.set_ylabel('percentage of permeability')
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        ax1.text(xmax*0.95, ymax*0.9, 'Max P: {:.4f}\nOverall P: {:.4f}'.format(df_norm['permeability'].max(), df_norm['permeability'].sum()), ha='right')

        fig2, ax2 = plt.subplots(1,1, figsize=(6,6))
        sns.barplot(data=df_exp, x='pores', y='permeability_percent', ax=ax2)
        ax2.set_ylabel('percentage of permeability')
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        ax2.text(xmax*0.95, ymax*0.9, 'Max P: {:.4f}\nOverall P: {:.4f}'.format(df_exp['permeability'].max(), df_exp['permeability'].sum()), ha='right')
        plt.show()


    if compare_effective_barriers:

        dist = 'normal'
        params = {'mu' : large_barrier, 'sigma' : sigma}
        model = Path(T=T, dist=dist, dist_params=params)
        dG_eff = model.calculate_effective_barrier() / (R*T)
        model.membrane_barriers = model.membrane_barriers / (R*T)
        ax = model.plot_distribution(bw='scott', show_hist=False, label=f'N({large_barrier/R/T:.0f}RT, {sigma/R/T:.0f}RT)')
        ax.axvline(dG_eff, ls='dashed', c='k', label='N: $\Delta G_{eff}$/RT = %.4f' % (dG_eff))

        dist = 'exponential'
        params = {'beta' : large_barrier}
        model = Path(T=T, dist=dist, dist_params=params)
        dG_eff = model.calculate_effective_barrier() / (R*T)
        model.membrane_barriers = model.membrane_barriers / (R*T)
        model.plot_distribution(bw='scott', show_hist=False, ax=ax, label=f'exp({large_barrier/R/T:.0f}RT)')
        ax.axvline(dG_eff, ls='dotted', c='k', label='exp: $\Delta G_{eff}$/RT = %.4f' % (dG_eff))
        
        ax.axvline(large_barrier/R/T, c='r', label='mean')
        ax.set_xlabel('$\Delta G_{M,j}$ / RT')
        plt.legend(loc='upper center')
        plt.show()
    
    if estimate_dH_dS:

        from tqdm import tqdm

        n_paths = 50
        dist = 'normal'
        params = {'mu' : large_barrier, 'sigma' : sigma}
        temps = [250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350]*10
        X = np.zeros(len(temps))
        Y = np.zeros(len(temps))

        print(f'Calculating permeabilities for {len(temps)} temperatures to estimate dH and dS')
        for i,T in tqdm(enumerate(temps)):
            model = EyringModel(T=T)
            for n in range(n_paths):
                model.add_Path(dist=dist, dist_params=params)

            P = model.calculate_permeability()
            lam = model.get_lambda()
            delta = np.array(model.deltas).mean()
            X[i] = 1 / T
            Y[i] = np.log(P*h*delta / (kB*T*lam**2))

        data = pd.DataFrame(np.array([X,Y]).T, columns=['1/T', 'ln(P h del / kB T lam^2)'])
        m, b = np.polyfit(X,Y,1)
        dH = -m*R
        dS = b*R

        sns.lmplot(x='1/T', y='ln(P h del / kB T lam^2)', data=data, scatter_kws={'alpha':0.75, 'edgecolor':'tab:blue'})
        plt.xlabel('1/T')
        plt.ylabel('ln($P h \delta$ / $k_B T \lambda$)')
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        plt.text(xmax*0.95, ymax*1.05, 'dH = {:.4f}\ndS = {:.4f}'.format(dH, dS), ha='right')
        plt.show()
        