# Script to run Eyring Model analyses

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from eyring_model import EyringModel, Path

# Define global constants
global kB 
kB = 1.380649 * 10**-23    # Boltzmann (m^2 kg / s^2 K)
global h
h = 6.62607 * 10**-34      # Planck (m^2 kg / s)
global R
R = 1.9858775 * 10**-3     # universal gas (kcal / mol K)

def parallel_pores(dH_barrier, dS_barrier, dH_sigma, dS_sigma, dG_barrier, T=300, multi=True):

    n_paths = 50
    n_jumps = 50
    fill = True

    print(f'\nCalculating effective barriers and fractions of permeability for {n_paths} paths through the membrane...')

    fig, ax = plt.subplots(3,1, figsize=(12,8), sharex=True)

    # ALL MEMBRANE BARRIERS EQUAL

    model_equal = EyringModel(T=T)
    dist = 'equal'
    params = {'mu' : np.array([dH_barrier, dS_barrier])}

    print(f'\tfor equal barriers:')

    # plot the membrane barrier distribution for each pore, overlapping
    effective_barriers = np.zeros(n_paths)
    for n in tqdm(range(n_paths)):
        model_equal.add_Path(n_jumps=n_jumps)
        model_equal.paths[n].generate_membrane_barriers(dist=dist, multi=multi, dist_params=params)
        effective_barriers[n] = model_equal.paths[n].calculate_effective_barrier()
        sns.histplot(model_equal.paths[n].membrane_barriers, edgecolor=None, ax=ax[0], stat='density', fill=fill, alpha=0.25)

    permeability = model_equal.calculate_permeability()
    effective_barrier_equal = model_equal.calculate_effective_barrier()

    # save data as pandas DataFrame
    df_equal = pd.DataFrame()
    df_equal['pores'] = np.arange(1,n_paths+1)
    df_equal['permeability'] = model_equal.permeabilities
    df_equal['effective_barriers'] = effective_barriers
    df_equal['permeability_percent'] = model_equal.permeabilities / model_equal.permeabilities.sum() * 100
    df_equal.sort_values('permeability_percent', ascending=False, inplace=True)
    df_equal['flux_fraction'] = df_equal['permeability_percent'].cumsum() / 100
    df_equal['pore_fraction'] = np.arange(1,n_paths+1) / n_paths
    df_equal.loc[len(df_equal.index)] = [0,0,0,0,0,0] # add zero row for ROC curve

    # NORMAL DISTRIBUTION OF BARRIERS

    model_norm = EyringModel(T=T)
    dist = 'normal'
    params = {'mu'  : np.array([dH_barrier, dS_barrier]),
              'cov' : np.array([[dH_sigma**2,0],
                                [0,dS_sigma**2]])}
    
    print(f'\tfor normal barriers:')

    # plot the membrane barrier distribution for each pore, overlapping
    effective_barriers = np.zeros(n_paths)
    for n in tqdm(range(n_paths)):
        model_norm.add_Path(n_jumps=n_jumps)
        model_norm.paths[n].generate_membrane_barriers(dist=dist, multi=multi, dist_params=params)
        effective_barriers[n] = model_norm.paths[n].calculate_effective_barrier()
        sns.histplot(model_norm.paths[n].membrane_barriers, binwidth=1, edgecolor=None, ax=ax[1], stat='density', fill=fill, alpha=0.25)

    permeability = model_norm.calculate_permeability()
    effective_barrier_norm = model_norm.calculate_effective_barrier()

    # save data as pandas DataFrame
    df_norm = pd.DataFrame()
    df_norm['pores'] = np.arange(1,n_paths+1)
    df_norm['permeability'] = model_norm.permeabilities
    df_norm['effective_barriers'] = effective_barriers
    df_norm['permeability_percent'] = model_norm.permeabilities / model_norm.permeabilities.sum() * 100
    df_norm.sort_values('permeability_percent', ascending=False, inplace=True)
    df_norm['flux_fraction'] = df_norm['permeability_percent'].cumsum() / 100
    df_norm['pore_fraction'] = np.arange(1,n_paths+1) / n_paths
    df_norm.loc[len(df_norm.index)] = [0,0,0,0,0,0] # add zero row for ROC curve
    

    # EXPONENTIAL DISTRIBUTION OF BARRIERS

    model_exp = EyringModel(T=T)
    dist = 'exponential'
    params = {'beta' : np.array([dH_barrier, dS_barrier])}

    print(f'\tfor exponential barriers:')

    # plot the membrane barrier distribution for each pore, overlapping
    effective_barriers = np.zeros(n_paths)
    for n in tqdm(range(n_paths)):
        model_exp.add_Path(n_jumps=n_jumps)
        model_exp.paths[n].generate_membrane_barriers(dist=dist, multi=multi, dist_params=params)
        effective_barriers[n] = model_exp.paths[n].calculate_effective_barrier()
        sns.histplot(model_exp.paths[n].membrane_barriers, binwidth=1, edgecolor=None, ax=ax[2], stat='density', fill=fill, alpha=0.25)

    permeability = model_exp.calculate_permeability()
    effective_barrier_exp = model_exp.calculate_effective_barrier()

    # save data as pandas DataFrame
    df_exp = pd.DataFrame()
    df_exp['pores'] = np.arange(1,n_paths+1)
    df_exp['permeability'] = model_exp.permeabilities
    df_exp['effective_barriers'] = effective_barriers
    df_exp['permeability_percent'] = model_exp.permeabilities / model_exp.permeabilities.sum() * 100
    df_exp.sort_values('permeability_percent', ascending=False, inplace=True)
    df_exp['flux_fraction'] = df_exp['permeability_percent'].cumsum() / 100
    df_exp['pore_fraction'] = np.arange(1,n_paths+1) / n_paths
    df_exp.loc[len(df_exp.index)] = [0,0,0,0,0,0] # add zero row for ROC curve
    
    # PLOTTING

    # plot the effective barrier, max barrier, and mean barrier
    ax[0].axvline(effective_barrier_equal, ls='dashed', c='k', label='effective barrier')
    ax[0].axvline(dG_barrier, ls='dashed', c='r', label='mean barrier')
    ax[0].legend()
    ax[0].set_title(f'All barriers equal, {dG_barrier:.4f}')

    ax[1].axvline(effective_barrier_norm, ls='dashed', c='k', label='effective barrier')
    ax[1].axvline(dG_barrier, ls='dashed', c='r', label='mean barrier')
    ax[1].legend()
    ax[1].set_title(f'Normal distribution, mean = {dG_barrier:.4f}, stdev = {effective_barrier_norm.std():.4f}')

    ax[2].axvline(effective_barrier_exp, ls='dashed', c='k', label='effective barrier')
    ax[2].axvline(dG_barrier, ls='dashed', c='r', label='mean barrier')
    ax[2].legend()
    ax[2].set_title(f'Exponential distribution, mean = {dG_barrier:.4f}')

    ax[2].set_xlabel('$\Delta G_{M,j}$')
    ax[2].set_xlim(-10,)

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
    
    fig3, ax3 = plt.subplots(1,1, figsize=(6,6))
    sns.lineplot(data=df_equal, x='pore_fraction', y='flux_fraction', ax=ax3, label='equal')
    sns.lineplot(data=df_norm, x='pore_fraction', y='flux_fraction', ax=ax3, label='normal')
    sns.lineplot(data=df_exp, x='pore_fraction', y='flux_fraction', ax=ax3, label='exponential')
    ax3.set_xlabel('fraction of the pores')
    ax3.set_ylabel('fraction of the flux')
    plt.show()


def compare_effective_barriers(dH_barrier, dS_barrier, dH_sigma, dS_sigma, dG_barrier, T=300, multi=True):

    print(f'\nComparing effective barriers for a single path through the membrane...')

    fig, ax1 = plt.subplots(1,2, figsize=(10,5), sharey=True, sharex=True)

    dist = 'normal'
    params = {'mu'  : np.array([dH_barrier, dS_barrier]),
              'cov' : np.array([[dH_sigma**2,0],
                                [0,dS_sigma**2]])}

    model = Path(T=T)
    model.generate_membrane_barriers(dist=dist, multi=multi, dist_params=params)
    dG_eff = model.calculate_effective_barrier()
    ax = model.plot_distribution(hist=True, color='tab:blue', binwidth=1, label='normal')
    ymin, ymax = plt.ylim()
    ax.axvline(dG_eff, ls='dashed', c='tab:blue')
    ax.text(dG_eff*1.1, ymax*0.9, '$\Delta G_{eff}$')
    ax1[0].scatter(-T*model.entropic_barriers, model.enthalpic_barriers, edgecolor='k', color='tab:blue')
    ax1[0].set_xlabel('$-T\Delta S$')
    ax1[0].set_ylabel('$\Delta H$')
    ax1[0].legend(['normal'])

    dist = 'exponential'
    params = {'beta'  : np.array([dH_barrier, dS_barrier])}
    model = Path(T=T)
    model.generate_membrane_barriers(dist=dist, multi=multi, dist_params=params)
    dG_eff = model.calculate_effective_barrier()
    model.plot_distribution(hist=True, color='tab:orange', binwidth=1, ax=ax, label='exponential')
    ymin, ymax = plt.ylim()
    ax.axvline(dG_eff, ls='dashed', c='tab:orange')
    ax.text(dG_eff*1.01, ymax*0.9, '$\Delta G_{eff}$')
    ax1[1].scatter(-T*model.entropic_barriers, model.enthalpic_barriers, edgecolor='k', color='tab:orange')
    ax1[1].set_xlabel('$-T\Delta S$')
    ax1[1].legend(['exponential'])
    
    ax.axvline(dG_barrier, c='r')
    ax.text(dG_barrier*0.5, ymax*0.9, 'mean', ha='left')
    ax.set_xlabel('$\Delta G_{M,j}$')
    plt.legend(loc='center')
    plt.show()

def plot_paths(dH_barrier, dS_barrier, dH_sigma, dS_sigma, T=300, multi=True):

    print(f'\nPlotting 4 realizations of barrier paths through the membrane...')

    fig, ax = plt.subplots(4,1, figsize=(8,20), sharex=True)

    for i in range(4):
    
        dist = 'normal'
        params = {
                'mu'  : np.array([dH_barrier, dS_barrier]),
                'cov' : np.array([[dH_sigma**2,0],
                                  [0,dS_sigma**2]])
                }
        model = Path(T=T)
        model.generate_membrane_barriers(dist=dist, multi=multi, dist_params=params)
        dG_eff = model.calculate_effective_barrier()
        ax[i].plot(model.jump_lengths.cumsum(), model.membrane_barriers, color='tab:blue', alpha=0.75, label='normal')
        ax[i].axhline(dG_eff, c='tab:blue', ls='dashed')
        ax[i].text(-5*model.lam, dG_eff*1.1, '$\Delta G_{eff}$', ha='right')

        dist = 'exponential'
        params = {'beta'  : np.array([dH_barrier, dS_barrier])}
        model = Path(T=T)
        model.generate_membrane_barriers(dist=dist, multi=multi, dist_params=params)
        dG_eff = model.calculate_effective_barrier()
        ax[i].plot(model.jump_lengths.cumsum(), model.membrane_barriers, color='tab:orange', alpha=0.75, label='exponential')
        ax[i].axhline(dG_eff, c='tab:orange', ls='dashed')
        ax[i].text(-5*model.lam, dG_eff*1.05, '$\Delta G_{eff}$', ha='right')

        ax[i].set_ylabel('$\Delta G_{M,j}$')
        ax[i].legend(loc='upper right')
        ax[i].set_ylim(0,dG_eff*1.5)
    
    ax[i].set_xlim(-20*model.lam,)
    ax[i].set_xlabel('membrane thickness (Angstroms)')
    plt.show()

def compare_jump_lengths(dH_barrier, dS_barrier, n_paths, delta=400, T=300, multi=True):

    dist = 'equal'
    params = params = {'mu'  : np.array([dH_barrier, dS_barrier])}

    lambdas = [1,2,3,4,5,6,7,8,9,10]*3 # list of jump lengths to compare

    print(f'\nComparing permeabilities and effective barriers for distributions of jump lengths with different means and with overall thickness {delta} Angstroms...')

    # Jump lengths EQUAL

    print(f'\tfor equal dsitribution:')

    jump_dist = 'equal'
    
    permeabilities = np.zeros(len(lambdas))
    deltas = np.zeros(len(lambdas))
    effective_barriers = np.zeros(len(lambdas))
        
    for i,lam in tqdm(enumerate(lambdas)):

        model = EyringModel(T=T)
        n_jumps = int(delta / lam)

        # add all parallel paths
        for n in range(n_paths):
            jump_params = {'mu' : lam}
            model.add_Path(n_jumps=n_jumps, lam=lam)
            model.paths[n].generate_membrane_barriers(dist=dist, multi=multi, dist_params=params)
            model.paths[n].generate_jump_distribution(dist=jump_dist, dist_params=jump_params)
    
            
        permeabilities[i] = model.calculate_permeability()
        deltas[i] = np.array(model.deltas).mean()
        effective_barriers[i] = model.calculate_effective_barrier()

    df1 = pd.DataFrame()
    df1['lambda'] = lambdas
    df1['permeability'] = permeabilities
    df1['thickness'] = deltas
    df1['effective_barriers'] = effective_barriers
    df1['distribution'] = ['equal']*len(lambdas)

    # Jump lengths NORMAL

    print(f'\tfor normal distribution:')
    
    jump_dist = 'normal'

    for i,lam in tqdm(enumerate(lambdas)):

        model = EyringModel(T=T)
        n_jumps = int(delta / lam)

        # add all parallel paths
        for n in range(n_paths):
            jump_params = {'mu' : lam, 'sigma' : lam/4}
            model.add_Path(n_jumps=n_jumps, lam=lam)
            model.paths[n].generate_membrane_barriers(dist=dist, multi=multi, dist_params=params)
            model.paths[n].generate_jump_distribution(dist=jump_dist, dist_params=jump_params)
                
        permeabilities[i] = model.calculate_permeability()
        deltas[i] = np.array(model.deltas).mean()
        effective_barriers[i] = model.calculate_effective_barrier()

    df2 = pd.DataFrame()
    df2['lambda'] = lambdas
    df2['permeability'] = permeabilities
    df2['thickness'] = deltas
    df2['effective_barriers'] = effective_barriers
    df2['distribution'] = ['normal']*len(lambdas)

    # Jump lengths EXPONENTIAL
    
    jump_dist = 'exponential'

    print(f'\tfor exponential distribution:')

    for i,lam in tqdm(enumerate(lambdas)):

        model = EyringModel(T=T)
        n_jumps = int(delta / lam)

        # add all parallel paths
        for n in range(n_paths):
            jump_params = {'beta' : lam}
            model.add_Path(n_jumps=n_jumps, lam=lam)
            model.paths[n].generate_membrane_barriers(dist=dist, multi=multi, dist_params=params)
            model.paths[n].generate_jump_distribution(dist=jump_dist, dist_params=jump_params)
                
        permeabilities[i] = model.calculate_permeability()
        deltas[i] = np.array(model.deltas).mean()
        effective_barriers[i] = model.calculate_effective_barrier()

    df3 = pd.DataFrame()
    df3['lambda'] = lambdas
    df3['permeability'] = permeabilities
    df3['thickness'] = deltas
    df3['effective_barriers'] = effective_barriers
    df3['distribution'] = ['exponential']*len(lambdas)

    sns.lineplot(data=df1, x='lambda', y='permeability', color='tab:blue', label='equal')
    sns.lineplot(data=df2, x='lambda', y='permeability', color='tab:orange', label='normal')
    sns.lineplot(data=df3, x='lambda', y='permeability', color='tab:green', label='exponential')
    plt.xlabel('mean jump length (Angstroms)')
    plt.ylabel('permeability ($L/m^2 h$)')
    plt.legend()
    plt.show()

    sns.scatterplot(data=df1, x='lambda', y='thickness', color='tab:blue', label='equal')
    sns.scatterplot(data=df2, x='lambda', y='thickness', color='tab:orange', label='normal')
    sns.scatterplot(data=df3, x='lambda', y='thickness', color='tab:green', label='exponential')
    plt.xlabel('mean jump length (Angstroms)')
    plt.ylabel('thickness (Angstroms)')
    plt.legend()
    plt.show()

    sns.scatterplot(data=df1, x='lambda', y='effective_barriers', color='tab:blue', label='equal')
    sns.scatterplot(data=df2, x='lambda', y='effective_barriers', color='tab:orange', label='normal')
    sns.scatterplot(data=df3, x='lambda', y='effective_barriers', color='tab:green', label='exponential')
    plt.xlabel('mean jump length (Angstroms)')
    plt.ylabel('$\Delta G_{eff}$')
    plt.legend()
    plt.show()

def estimate_dH_dS(dH_barrier, dS_barrier, dH_sigma, dS_sigma, n_paths, plot=False):

    print(f'\nEstimating the effective enthalpic and entropic barriers...')

    multi = True

    temps = np.array([250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350]*3)

    dG_eff = np.zeros(len(temps))
    P = np.zeros(len(temps))
    X = np.zeros(len(temps))
    Y = np.zeros(len(temps))

    fig, ax = plt.subplots(2,2, figsize=(10,10), sharex=True)

    # MULTIVARIATE NORMAL

    print('\tfor normal distributions:')

    params = {
        'mu'  : np.array([dH_barrier, dS_barrier]),
        'cov' : np.array([[dH_sigma**2,0],
                          [0,dS_sigma**2]])
    }

    dist = 'normal'

    dH = 0
    dS = 0

    for i, T in tqdm(enumerate(temps)):
        model = EyringModel(T=T)
        for n in range(n_paths):
            model.add_Path(n_jumps=200, lam=10)
            model.paths[n].generate_membrane_barriers(dist=dist, multi=multi, dist_params=params)
            dH += model.paths[n].enthalpic_barriers.mean()
            dS += model.paths[n].entropic_barriers.mean()
            if plot and T == 300:
                sns.histplot(model.paths[n].enthalpic_barriers, edgecolor=None, ax=ax[0,0], stat='density', alpha=0.1, color='cornflowerblue')
                sns.histplot(-300*model.paths[n].entropic_barriers, edgecolor=None, ax=ax[0,1], stat='density', alpha=0.1, color='lightcoral')

        P[i] = model.calculate_permeability() / 60 / 60 / 1000 * 10**9 * 10
        dG_eff[i] = model.calculate_effective_barrier()
        lam = model.get_lambda()
        delta = np.array(model.deltas).mean()
        X[i] = 1 / T
        Y[i] = np.log(P[i]*h*delta / (kB*T*lam**2))

    dHm = model.paths[n].enthalpic_barriers.mean()
    dSm = model.paths[n].entropic_barriers.mean()
    print(f'\nSingle path dH: {dHm}')
    print(f'Single path dS: {dSm} or -T dS at 300 K: {-300*dSm}')
    print(f'Many path contribution R ln(n): {R*np.log(n_paths)} or -RT ln(n) at 300 K: {-R*300*np.log(n_paths)}')

    avg_dH = dH / n_paths / len(temps)
    avg_dS = dS / n_paths / len(temps)
    print(f'\nAverage dH: {avg_dH}')
    print(f'Average dS: {avg_dS}')

    A = np.vstack([X, np.ones(len(X))]).T
    m, b = np.linalg.lstsq(A,Y, rcond=None)[0]
    print(f'\ndH_eff : {-m*R}')
    print(f'dS_eff : {b*R} or -T dS_eff at 300 K: {-300*b*R}')
    print(f'dG_eff at 300 K: {dG_eff.mean()}')

    if plot:
        # plot effective, single path, mean barriers
        ax[0,0].set_title('$\Delta H$ normal')
        ax[0,1].set_title('$-T\Delta S$ normal')

        ax[0,0].axvline(-m*R, ls='dashed', c='blue', label='effective')
        ax[0,0].axvline(avg_dH, ls='dashed', c='red', label='mean')
        ax[0,0].axvline(dG_eff.mean(), ls='dashed', c='k', label='$\Delta G_{eff}$')

        ax[0,1].axvline(-300*b*R, ls='dashed', c='blue', label='effective')
        ax[0,1].axvline(-300*avg_dS, ls='dashed', c='red', label='mean')
        ax[0,1].axvline(dG_eff.mean(), ls='dashed', c='k', label='$\Delta G_{eff}$')

        ax[0,1].legend()

    df1 = pd.DataFrame()
    df1['distribution'] = ['multi-variate normal']*len(temps)
    df1['temperature'] = temps
    df1['permeability'] = P
    df1['effective free energy'] = dG_eff
    df1['1/T'] = X
    df1['ln(P h del / kB T lam^2)'] = Y

    # MULTIPLE EXPONENTIALS

    print('\tfor exponential distributions:')

    params = {'beta'  : np.array([dH_barrier, dS_barrier])}

    dist = 'exponential'

    dH = 0
    dS = 0

    for i, T in tqdm(enumerate(temps)):
        model = EyringModel(T=T)
        for n in range(n_paths):
            model.add_Path(n_jumps=200, lam=10)
            model.paths[n].generate_membrane_barriers(dist=dist, multi=multi, dist_params=params)
            dH += model.paths[n].enthalpic_barriers.mean()
            dS += model.paths[n].entropic_barriers.mean()
            if plot and T == 300:
                sns.histplot(model.paths[n].enthalpic_barriers, edgecolor=None, ax=ax[1,0], stat='density', alpha=0.1, color='cornflowerblue')
                sns.histplot(-300*model.paths[n].entropic_barriers, edgecolor=None, ax=ax[1,1], stat='density', alpha=0.1, color='lightcoral')

        dG_eff[i] = model.calculate_effective_barrier()
        P[i] = model.calculate_permeability()
        lam = model.get_lambda()
        delta = np.array(model.deltas).mean()
        X[i] = 1 / T
        Y[i] = np.log(P[i]*h*delta / (kB*T*lam**2))

    dHm = model.paths[n].enthalpic_barriers.mean()
    dSm = model.paths[n].entropic_barriers.mean()
    print(f'\nSingle path dH: {dHm}')
    print(f'Single path dS: {dSm} or -T dS at 300 K: {-300*dSm}')
    print(f'Many path contribution R ln(n): {R*np.log(n_paths)} or -RT ln(n) at 300 K: {-R*300*np.log(n_paths)}')

    avg_dH = dH / n_paths / len(temps)
    avg_dS = dS / n_paths / len(temps)
    print(f'\nAverage dH: {avg_dH}')
    print(f'Average dS: {avg_dS}')

    A = np.vstack([X, np.ones(len(X))]).T
    m, b = np.linalg.lstsq(A,Y, rcond=None)[0]
    print(f'\ndH_eff : {-m*R}')
    print(f'dS_eff : {b*R} or -T dS_eff at 300 K: {-300*b*R}')
    print(f'dG_eff at 300 K: {dG_eff.mean()}')

    df2 = pd.DataFrame()
    df2['distribution'] = ['multiple exponentials']*len(temps)
    df2['temperature'] = temps
    df2['permeability'] = P
    df2['effective free energy'] = dG_eff
    df2['1/T'] = X
    df2['ln(P h del / kB T lam^2)'] = Y

    if plot:
        # plot effective, single path, mean barriers
        ax[1,0].set_title('$\Delta H$ exponential')
        ax[1,1].set_title('$-T\Delta S$ exponential')

        ax[1,0].axvline(-m*R, ls='dashed', c='blue', label='effective')
        ax[1,0].axvline(avg_dH, ls='dashed', c='red', label='mean')
        ax[1,0].axvline(dG_eff.mean(), ls='dashed', c='k', label='$\Delta G_{eff}$')

        ax[1,1].axvline(-300*b*R, ls='dashed', c='blue', label='effective')
        ax[1,1].axvline(-300*avg_dS, ls='dashed', c='red', label='mean')
        ax[1,1].axvline(dG_eff.mean(), ls='dashed', c='k', label='$\Delta G_{eff}$')

        ax[1,1].legend()

    df = pd.concat((df1,df2))

    sns.lmplot(data=df, x='1/T', y='ln(P h del / kB T lam^2)', hue='distribution', 
               scatter_kws={'alpha':0.75, 'edgecolor':'black'})

    plt.figure()
    sns.scatterplot(data=df, x='temperature', y='permeability', hue='distribution')

    sns.lmplot(data=df, x='temperature', y='effective free energy', hue='distribution',
               scatter_kws={'alpha':0.75, 'edgecolor':'black'})

    plt.show()


def show_maximums(dH_barrier, dS_barrier, dH_sigma, dS_sigma, T=300, multi=True):

    print(f'\nShowing maximum barriers across parallel paths...')

    n_paths = 2000

    fig, ax = plt.subplots(2,1, figsize=(8,10), sharex=True)

    # NORMAL DISTRIBUTION OF BARRIERS

    model = EyringModel(T=T)
    dist = 'normal'
    params = {'mu'  : np.array([dH_barrier, dS_barrier]),
              'cov' : np.array([[dH_sigma**2,0],
                                [0,dS_sigma**2]])}
    
    print(f'\tfor normal barriers:')

    # plot the membrane barrier distribution for each pore, overlapping
    max_barriers = np.zeros(n_paths)
    for n in tqdm(range(n_paths)):
        model.add_Path()
        model.paths[n].generate_membrane_barriers(dist=dist, multi=multi, dist_params=params)
        max_barriers[n] = model.paths[n].membrane_barriers.max()

    effective_barrier = model.calculate_effective_barrier()
    shifted_effective_barrier = effective_barrier + R*T*np.log(n_paths)

    paths = np.arange(1, n_paths+1)
    ax[0].scatter(paths, max_barriers, edgecolors='k')
    ax[0].axhline(effective_barrier, ls='dashed', c='k')
    ax[0].axhline(shifted_effective_barrier, ls='dashed', c='r')
    xmin, xmax = ax[0].get_xlim()
    ymin, ymax = ax[0].get_ylim()
    ax[0].text(xmax*0.75, effective_barrier-1, '$\Delta G_{eff}$')
    ax[0].text(xmax*0.75, shifted_effective_barrier-1, '$\Delta G_{eff} + RT \ln(N)$', c='r')
    ax[0].set_ylabel('$\Delta G_{max}$')
    ax[0].set_ylim(ymin-3, ymax)
    
    # EXPONENTIAL DISTRIBUTION OF BARRIERS

    model = EyringModel(T=T)
    dist = 'exponential'
    params = {'beta'  : np.array([dH_barrier, dS_barrier])}
    
    print(f'\tfor exponential barriers:')

    # plot the membrane barrier distribution for each pore, overlapping
    max_barriers = np.zeros(n_paths)
    for n in tqdm(range(n_paths)):
        model.add_Path()
        model.paths[n].generate_membrane_barriers(dist=dist, multi=multi, dist_params=params)
        max_barriers[n] = model.paths[n].membrane_barriers.max()

    effective_barrier = model.calculate_effective_barrier()
    shifted_effective_barrier = effective_barrier + R*T*np.log(n_paths)

    paths = np.arange(1, n_paths+1)
    ax[1].scatter(paths, max_barriers, edgecolors='k', c='tab:orange')
    ax[1].axhline(effective_barrier, ls='dashed', c='k')
    ax[1].axhline(shifted_effective_barrier, ls='dashed', c='r')
    xmin, xmax = ax[1].get_xlim()
    ymin, ymax = ax[1].get_ylim()
    ax[1].text(xmax*0.75, effective_barrier-5, '$\Delta G_{eff}$')
    ax[1].text(xmax*0.75, shifted_effective_barrier-3, '$\Delta G_{eff} + RT \ln(N)$', c='r')
    ax[1].set_ylabel('$\Delta G_{max}$')
    ax[1].set_ylim(ymin-10, ymax)
    
    ax[1].set_xlabel('paths')
    plt.show()


def fixed_jump_length(dH_barrier, dS_barrier, n_paths=2000, T=300, multi=True):

    dist = 'equal'
    params = params = {'mu'  : np.array([dH_barrier, dS_barrier])}

    lam = 10 # fixed 10 Angstrom jump length
    n_jumps = np.array([10,20,30,40,50,100,200,300,400,500,1000]) # changing number of jumps
    
    permeabilities = np.zeros(len(n_jumps))
    deltas = np.zeros(len(n_jumps))
    effective_barriers = np.zeros(len(n_jumps))
        
    for i,nj in tqdm(enumerate(n_jumps)):

        model = EyringModel(T=T)

        # add all parallel paths
        for n in range(n_paths):
            model.add_Path(n_jumps=nj, lam=lam)
            model.paths[n].generate_membrane_barriers(dist=dist, multi=multi, dist_params=params)
    
        permeabilities[i] = model.calculate_permeability()
        deltas[i] = np.array(model.deltas).mean()
        effective_barriers[i] = model.calculate_effective_barrier()

    df = pd.DataFrame()
    df['jumps'] = n_jumps
    df['permeability'] = permeabilities
    df['thickness'] = deltas
    df['effective_barriers'] = effective_barriers

    sns.scatterplot(data=df, x='jumps', y='effective_barriers')
    plt.show()

    sns.scatterplot(data=df, x='thickness', y='permeability')
    plt.show()


def barrier_variance(dH_barrier, dS_barrier, n_paths=2000, T=300):

    multi = True
    dist = 'normal'

    sigs = np.array([0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 10])
    # dH_sigs = np.array([0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 10])
    # dS_sigs = np.array([10e-5, 10e-4, 5e-4, 1e-4, 5e-3, 1e-3, 0.05, 0.01, 0.1, 0.5, 1, 2, 3])

    # save data per path for ROC curves
    perm_per_path = np.zeros(n_paths*len(sigs)**2)
    perm_percent = np.zeros(n_paths*len(sigs)**2)
    models_dH = np.zeros(n_paths*len(sigs)**2)
    models_dS = np.zeros(n_paths*len(sigs)**2)
    models = np.zeros(n_paths*len(sigs)**2)

    # save data for overall model
    effective_barriers = np.zeros(len(sigs)**2)
    permeabilities = np.zeros(len(sigs)**2)
    max_barriers = np.zeros(len(sigs)**2)
    max_enthalpies = np.zeros(len(sigs)**2)
    max_entropies = np.zeros(len(sigs)**2)
    dH_sigmas = np.zeros(len(sigs)**2)
    dS_sigmas = np.zeros(len(sigs)**2)
    i = 0
    for dH_sig in tqdm(sigs):
        for dS_sig in sigs:

            model = EyringModel(T=T)
            params = {'mu'  : np.array([dH_barrier, dS_barrier]),
                    'cov' : np.array([[dH_sig**2,0],
                                        [0,dS_sig**2]])}
    
            dH_max = -10e8
            dS_max = -10e8
            dG_max = -10e8
            for n in range(n_paths):
                model.add_Path(lam=10)
                model.paths[n].generate_membrane_barriers(dist=dist, multi=multi, dist_params=params)
                dH_max = max(dH_max, model.paths[n].enthalpic_barriers.max())
                dS_max = max(dS_max, model.paths[n].entropic_barriers.max())
                dG_max = max(dG_max, model.paths[n].membrane_barriers.max())

            dH_sigmas[i] = dH_sig
            dS_sigmas[i] = dS_sig
            effective_barriers[i] = model.calculate_effective_barrier()
            permeabilities[i] = model.calculate_permeability()
            max_barriers[i] = dG_max
            max_enthalpies[i] = dH_max
            max_entropies[i] = dS_max
            perm_per_path[i*n_paths:(i+1)*n_paths] = model.permeabilities
            perm_percent[i*n_paths:(i+1)*n_paths] = model.permeabilities / model.permeabilities.sum() * 100
            models_dH[i*n_paths:(i+1)*n_paths] = dH_sig
            models_dS[i*n_paths:(i+1)*n_paths] = dS_sig
            models[i*n_paths:(i+1)*n_paths] = i+1
            i += 1

    df_roc = pd.DataFrame()
    df_roc['paths'] = np.arange(1,n_paths+1).tolist()*len(sigs)**2
    df_roc['permeability'] = perm_per_path
    df_roc['permeability percent'] = perm_percent
    df_roc['dH sigma'] = models_dH
    df_roc['dS sigma'] = models_dS
    df_roc['model'] = models
    df_roc.to_csv('barrier_variance_ROC.csv', index=False)

    df = pd.DataFrame()
    df['dH sigma'] = dH_sigmas
    df['dS sigma'] = dS_sigmas
    df['effective barrier'] = effective_barriers
    df['permeability'] = permeabilities
    df['max barrier'] = max_barriers
    df['max enthalpic barrier'] = max_enthalpies
    df['max entropic barrier'] = max_entropies
    df.to_csv('barrier_variance.csv', index=False)


if __name__ == '__main__':

    # Inputs for testing barriers
    T = 300
    multi = True
    dH_barrier = 4.5
    dS_barrier = -6/300
    dH_sigma = 1.5
    dS_sigma = 2/300
    n_paths = 2000 # infinite limit

    dG_barrier = dH_barrier - T*dS_barrier

    # Choose what analyses to run
    # parallel_pores(dH_barrier, dS_barrier, dH_sigma, dS_sigma, dG_barrier, T=T, multi=multi)
    # compare_effective_barriers(dH_barrier, dS_barrier, dH_sigma, dS_sigma, dG_barrier, T=T, multi=multi)
    # plot_paths(dH_barrier, dS_barrier, dH_sigma, dS_sigma, T=T, multi=multi)
    # compare_jump_lengths(dH_barrier, dS_barrier, n_paths, delta=400, T=T, multi=multi)
    # estimate_dH_dS(dH_barrier, dS_barrier, dH_sigma, dS_sigma, n_paths)
    # estimate_dH_dS(dH_barrier, dS_barrier, dH_sigma, dS_sigma, n_paths=50, plot=True)
    # show_maximums(dH_barrier, dS_barrier, dH_sigma, dS_sigma, T=T, multi=multi)
    # fixed_jump_length(dH_barrier, dS_barrier, n_paths=n_paths, T=T, multi=multi)
    barrier_variance(dH_barrier, dS_barrier, n_paths=1200, T=T)
