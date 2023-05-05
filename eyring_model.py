# EyringModel class

import numpy as np
import matplotlib.pyplot as plt
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
                 barrier_sm=barrier_sm*R*self.T, barrier_ms=barrier_ms*R*self.T) # interfacial barriers
        
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

        self.effective_barrier = -R*self.T * np.log(delta / lam**2 * h / kB / self.T * self.permeabilities.sum() * 10**10 / 1000 / 60 / 60)

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
        self.enthalpic_barriers = None
        self.entropic_barriers = None

        # Default values for some parameters
        self.generate_membrane_barriers(dist=dist, dist_params=dist_params)
        self.generate_jump_distribution(dist_params={'mu' : lam}) # jump length distributions
        self.lam_sm = lam                                # solution-membrane jump length
        self.lam_ms = lam                                # membrane-solution jump length
        
        self.barrier_sm = barrier_sm             # solution-membrane barrier
        self.barrier_ms = barrier_ms             # membrane-solution barrier


    def generate_membrane_barriers(self, dist='normal', dist_params={'mu': 0, 'sigma' : 1}, multi=False, seed=None):
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

        if multi and dist in ['normal', 'N', 'norm']: # multivariate normal distribution of barriers
            # Raise an error if the correct parameters are not provided
            if 'mu' not in dist_params.keys():
                raise DistributionError("'mu' must be a key in distribution parameters for a multivariate normal distribution")
            elif 'cov' not in dist_params.keys():
                raise DistributionError("'cov' must be a key in distribution parameters for a multivariate normal distribution")
            
            # generate enthalpy and entropy distributions
            multi_norm = rng.multivariate_normal(mean=dist_params['mu'], cov=dist_params['cov'], size=self.n_jumps)
            self.enthalpic_barriers = multi_norm[:,0]
            self.entropic_barriers = -multi_norm[:,1] / self.T
            self.membrane_barriers = self.enthalpic_barriers - self.T*self.entropic_barriers # calculate dG from dH and dS

            # plt.hist(self.enthalpic_barriers, color='r', alpha=0.5, edgecolor='k')
            # plt.hist(-self.T*self.entropic_barriers, color='b', alpha=0.5, edgecolor='k')
            # plt.show()

        elif multi and dist in ['exponential', 'exp']: # multiple exponential distributions of barriers
            # Raise an error if the correct parameters are not provided
            if 'beta' not in dist_params.keys():
                raise DistributionError("'beta' must be a key in distribution parameters for an exponential distribution")

            # NOTE: assuming no covariance here, so drawing from two independent exponential distributions with means beta

            # generate enthalpy and entropy distributions
            self.enthalpic_barriers = rng.exponential(scale=dist_params['beta'][0], size=self.n_jumps)
            self.entropic_barriers = -rng.exponential(scale=dist_params['beta'][1], size=self.n_jumps)
            self.membrane_barriers = self.enthalpic_barriers - self.T*self.entropic_barriers # calculate dG

        elif dist in ['normal', 'N', 'norm']: # normal distribution of barriers
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


        if not multi:
            return self.membrane_barriers
        else:
            return self.membrane_barriers, self.enthalpic_barriers, self.entropic_barriers

    
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

        A = h / (kB*temp) / 60 / 60
        lam = self.jump_lengths / 10**10
        exp = np.exp(- self.membrane_barriers / (R*temp))
    
        return A*np.sum( 1 / (lam * exp) ) # units = h / m


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


    def plot_distribution(self, bw=None, hist=False, n_bins=50, binwidth=None, label=None, fill=True, ax=None, color=None):

        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=(10,6))

        if hist:
            sns.histplot(self.membrane_barriers, bins=n_bins, binwidth=binwidth, edgecolor='black',
                         stat='density', ax=ax, color=color, alpha=0.5, label=label)
        else:
            sns.kdeplot(self.membrane_barriers, bw_method=bw, label=label, fill=fill, ax=ax, color=color)
        
        plt.xlabel('$\Delta G_{M,j}$')

        return ax


if __name__ == '__main__':

    T = 300
    lam = 10
    n_jumps = 50
    multi = True
    dist = 'normal'
    dist_params = {'mu' : np.array([14,8]),
                   'cov': np.array([[14/3,0],
                                    [0,8/3]])}

    p = Path(T=T, n_jumps=n_jumps, lam=lam)
    dG, dH = p.generate_membrane_barriers(dist=dist, dist_params=dist_params, multi=multi)

    fig, ax = plt.subplots(2,1, figsize=(6,10), sharex=True)
    ax[0].hist(dG, bins=50, edgecolor='k', alpha=0.75)
    ax[0].set_xlabel('dG')
    ax[1].hist(dH, bins=50, edgecolor='k', alpha=0.75)
    ax[1].set_xlabel('dH')
    plt.show()