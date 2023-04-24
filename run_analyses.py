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

# Choose what analyses to run
parallel_pores = True
compare_effective_barriers = False
estimate_dH_dS_barrier_distributions = False
estimate_dH_dS_jump_distributions = False
compare_jump_lengths = False
plot_paths = False

# Inputs for testing barriers
T = 300
large = 18
large_barrier = large*R*T
small_barrier = large_barrier / 2
sigma = large_barrier / 3

if parallel_pores:

    n_paths = 50
    fill = True

    fig, ax = plt.subplots(3,1, figsize=(12,8), sharex=True)

    # ALL MEMBRANE BARRIERS EQUAL

    model_equal = EyringModel(T=T)
    dist = 'equal'
    params = {'mu' : large_barrier}

    # plot the membrane barrier distribution for each pore, overlapping
    effective_barriers = np.zeros(n_paths)
    for n in range(n_paths):
        model_equal.add_Path(dist=dist, dist_params=params)
        effective_barriers[n] = model_equal.paths[n].calculate_effective_barrier() / (R*T)
        sns.histplot(model_equal.paths[n].membrane_barriers / (R*T), ax=ax[0], stat='density', fill=fill)
        # model_equal.paths[n].plot_distribution(fill=fill, ax=ax[0])

    permeability = model_equal.calculate_permeability()
    effective_barrier_equal = model_equal.calculate_effective_barrier() / (R*T)

    # save data as pandas DataFrame
    df_equal = pd.DataFrame()
    df_equal['pores'] = np.arange(1,n_paths+1)
    df_equal['permeability'] = model_equal.permeabilities
    df_equal['effective_barriers'] = effective_barriers
    df_equal['permeability_percent'] = model_equal.permeabilities / model_equal.permeabilities.sum() * 100

    # NORMAL DISTRIBUTION OF BARRIERS

    model_norm = EyringModel(T=T)
    dist = 'normal'
    params = {'mu' : large_barrier, 'sigma' : sigma}

    # plot the membrane barrier distribution for each pore, overlapping
    effective_barriers = np.zeros(n_paths)
    for n in range(n_paths):
        model_norm.add_Path(dist=dist, dist_params=params)
        effective_barriers[n] = model_norm.paths[n].calculate_effective_barrier() / (R*T)
        sns.histplot(model_norm.paths[n].membrane_barriers / (R*T), ax=ax[1], stat='density', fill=fill)
        # model_norm.paths[n].plot_distribution(fill=fill, ax=ax[1])

    permeability = model_norm.calculate_permeability()
    effective_barrier_norm = model_norm.calculate_effective_barrier() / (R*T)

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
        sns.histplot(model_exp.paths[n].membrane_barriers / (R*T), ax=ax[2], stat='density', fill=fill)
        # model_exp.paths[n].plot_distribution(fill=fill, ax=ax[2])

    permeability = model_exp.calculate_permeability()
    effective_barrier_exp = model_exp.calculate_effective_barrier() / (R*T)

    # save data as pandas DataFrame
    df_exp = pd.DataFrame()
    df_exp['pores'] = np.arange(1,n_paths+1)
    df_exp['permeability'] = model_exp.permeabilities
    df_exp['effective_barriers'] = effective_barriers
    df_exp['permeability_percent'] = model_exp.permeabilities / model_exp.permeabilities.sum() * 100

    # PLOTTING

    # plot the effective barrier, max barrier, and mean barrier
    ax[0].axvline(effective_barrier_equal, ls='dashed', c='k', label='effective barrier')
    ax[0].axvline(large_barrier/R/T, ls='dashed', c='r', label='mean barrier')
    ax[0].legend()
    ax[0].set_title(f'All barriers equal, {large_barrier/R/T:.0f}RT')

    ax[1].axvline(effective_barrier_norm, ls='dashed', c='k', label='effective barrier')
    ax[1].axvline(large_barrier/R/T, ls='dashed', c='r', label='mean barrier')
    ax[1].legend()
    ax[1].set_title(f'Normal distribution, mean = {large_barrier/R/T:.0f}RT, stdev = {sigma/R/T:.0f}RT')

    ax[2].axvline(effective_barrier_exp, ls='dashed', c='k', label='effective barrier')
    ax[2].axvline(large_barrier/R/T, ls='dashed', c='r', label='mean barrier')
    ax[2].legend()
    ax[2].set_title(f'Exponential distribution, mean = {large_barrier/R/T:.0f}RT')

    ax[2].set_xlabel('$\Delta G_{M,j} / RT$')

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

if estimate_dH_dS_barrier_distributions:

    n_paths = 50
    temps = [250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350]*10
    X = np.zeros(len(temps))
    Y = np.zeros(len(temps))
    dG = np.zeros(len(temps))

    # ALL MEMBRANE BARRIERS EQUAL

    dist = 'equal'

    print(f'Calculating permeabilities for {len(temps)} temperatures to estimate dH and dS for {dist} distribution of barriers')
    for i,T in tqdm(enumerate(temps)):
        model = EyringModel(T=T)
        for n in range(n_paths):
            model.add_Path(dist=dist, dist_params={'mu' : large*R*300}) # membrane barrier distribution parameters change for each T

        dG[i] = model.calculate_effective_barrier()
        P = model.calculate_permeability()
        lam = model.get_lambda()
        delta = np.array(model.deltas).mean()
        X[i] = 1 / T
        Y[i] = np.log(P*h*delta / (kB*T*lam**2))

    df1 = pd.DataFrame(np.array([X,Y]).T, columns=['1/T', 'ln(P h del / kB T lam^2)'])
    df1['distribution'] = ['equal']*len(temps)
    df1['dG_eff'] = dG
    df1['T'] = temps
    m, b = np.polyfit(X,Y,1)
    dH_equal = -m*R
    dS_equal = b*R

    # NORMAL DISTRIBUTIONS OF BARRIERS

    dist = 'normal'

    print(f'Calculating permeabilities for {len(temps)} temperatures to estimate dH and dS for {dist} distribution of barriers')
    for i,T in tqdm(enumerate(temps)):
        model = EyringModel(T=T)
        for n in range(n_paths):
            model.add_Path(dist=dist, dist_params={'mu' : large*R*300, 'sigma' : large*R*300/3})

        dG[i] = model.calculate_effective_barrier()
        P = model.calculate_permeability()
        lam = model.get_lambda()
        delta = np.array(model.deltas).mean()
        X[i] = 1 / T
        Y[i] = np.log(P*h*delta / (kB*T*lam**2))

    df2 = pd.DataFrame(np.array([X,Y]).T, columns=['1/T', 'ln(P h del / kB T lam^2)'])
    df2['distribution'] = ['normal']*len(temps)
    df2['dG_eff'] = dG
    df2['T'] = temps
    m, b = np.polyfit(X,Y,1)
    dH_norm = -m*R
    dS_norm = b*R

    # EXPONENTIAL DISTRIBUTIONS OF BARRIERS

    dist = 'exponential'

    print(f'Calculating permeabilities for {len(temps)} temperatures to estimate dH and dS for {dist} distribution of barriers')
    for i,T in tqdm(enumerate(temps)):
        model = EyringModel(T=T)
        for n in range(n_paths):
            model.add_Path(dist=dist, dist_params={'beta' : large*R*300})

        dG[i] = model.calculate_effective_barrier()
        P = model.calculate_permeability()
        lam = model.get_lambda()
        delta = np.array(model.deltas).mean()
        X[i] = 1 / T
        Y[i] = np.log(P*h*delta / (kB*T*lam**2))

    df3 = pd.DataFrame(np.array([X,Y]).T, columns=['1/T', 'ln(P h del / kB T lam^2)'])
    df3['distribution'] = ['exponential']*len(temps)
    df3['dG_eff'] = dG
    df3['T'] = temps
    m, b = np.polyfit(X,Y,1)
    dH_exp = -m*R
    dS_exp = b*R

    data = pd.concat((df1,df2,df3))

    sns.lmplot(x='1/T', y='ln(P h del / kB T lam^2)', data=data, hue='distribution', scatter_kws={'alpha':0.75, 'edgecolor':'black'})
    plt.xlabel('1/T')
    plt.ylabel('ln($P h \delta$ / $k_B T \lambda$)')
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    # plt.text(xmax*0.95, ymax*1.05, 'dH = {:.4f}\ndS = {:.4f}'.format(dH, dS), ha='right')
    print('\nFor equal barriers = {:.0f}RT: \ndH = {:.4f} kcal/mol\ndS = {:.4f} kcal/mol'.format(large, dH_equal, dS_equal))
    print('\nFor normally distributed barriers with mean = {:.0f}RT: \ndH = {:.4f} kcal/mol\ndS = {:.4f} kcal/mol'.format(large, dH_norm, dS_norm))
    print('\nFor exponentially distributed barriers with mean = {:.0f}RT: \ndH = {:.4f}kcal/mol\ndS = {:.4f} kcal/mol'.format(large, dH_exp, dS_exp))
    plt.show()
    
    sns.lmplot(data=data, x='T', y='dG_eff', hue='distribution', scatter_kws={'alpha':0.75, 'edgecolor':'black'})
    plt.show()

if plot_paths:

    fig, ax = plt.subplots(4,1, figsize=(8,20), sharex=True)

    for i in range(4):
    
        dist = 'normal'
        params = {'mu' : large_barrier, 'sigma' : sigma}
        model = Path(T=T, dist=dist, dist_params=params)
        dG_eff = model.calculate_effective_barrier() / (R*T)
        model.membrane_barriers = model.membrane_barriers / (R*T)
        ax[i].plot(model.jump_lengths.cumsum(), model.membrane_barriers, color='tab:blue', alpha=0.75, label='normal')
        ax[i].axhline(dG_eff, c='tab:blue', ls='dashed')
        ax[i].text(-5*model.lam, dG_eff*1.1, '$\Delta G_{eff}$', ha='right')

        dist = 'exponential'
        params = {'beta' : large_barrier}
        model = Path(T=T, dist=dist, dist_params=params)
        dG_eff = model.calculate_effective_barrier() / (R*T)
        model.membrane_barriers = model.membrane_barriers / (R*T)
        ax[i].plot(model.jump_lengths.cumsum(), model.membrane_barriers, color='tab:orange', alpha=0.75, label='exponential')
        ax[i].axhline(dG_eff, c='tab:orange', ls='dashed')
        ax[i].text(-5*model.lam, dG_eff*1.05, '$\Delta G_{eff}$', ha='right')

        ax[i].set_ylabel('$\Delta G_{M,j}$ / RT')
        ax[i].legend(loc='upper right')
        ax[i].set_ylim(0,dG_eff*1.5)
    
    ax[i].set_xlim(-20*model.lam,)
    ax[i].set_xlabel('membrane thickness (Angstroms)')
    plt.show()

if compare_jump_lengths:

    n_paths = 50

    dist = 'equal'
    params = {'mu' : large_barrier}

    delta = 400 # set a fixed thickness
    lambdas = [1,2,3,4,5,6,7,8,9,10]*10 # list of jump lengths to compare

    # Jump lengths EQUAL

    jump_dist = 'equal'
    
    permeabilities = np.zeros(len(lambdas))
    effective_barriers = np.zeros(len(lambdas))
        
    for i,lam in tqdm(enumerate(lambdas)):

        model = EyringModel(T=T)
        n_jumps = int(delta / lam)

        # add all parallel paths
        for n in range(n_paths):
            jump_params = {'mu' : lam}
            model.add_Path(dist=dist, dist_params=params, n_jumps=n_jumps, lam=lam)
            model.paths[n].generate_jump_distribution(dist=jump_dist, dist_params=jump_params)
    
            
        permeabilities[i] = model.calculate_permeability()

    df1 = pd.DataFrame()
    df1['lambda'] = lambdas
    df1['permeability'] = permeabilities
    df1['distribution'] = ['equal']*len(lambdas)

    # Jump lengths NORMAL
    
    jump_dist = 'normal'

    for i,lam in tqdm(enumerate(lambdas)):

        model = EyringModel(T=T)
        n_jumps = int(delta / lam)

        # add all parallel paths
        for n in range(n_paths):
            jump_params = {'mu' : lam, 'sigma' : lam/4}
            model.add_Path(dist=dist, dist_params=params, n_jumps=n_jumps, lam=lam)
            model.paths[n].generate_jump_distribution(dist=jump_dist, dist_params=jump_params)
                
        permeabilities[i] = model.calculate_permeability()

    df2 = pd.DataFrame()
    df2['lambda'] = lambdas
    df2['permeability'] = permeabilities
    df2['distribution'] = ['normal']*len(lambdas)

    # Jump lengths EXPONENTIAL
    
    jump_dist = 'exponential'

    for i,lam in tqdm(enumerate(lambdas)):

        model = EyringModel(T=T)
        n_jumps = int(delta / lam)

        # add all parallel paths
        for n in range(n_paths):
            jump_params = {'beta' : lam}
            model.add_Path(dist=dist, dist_params=params, n_jumps=n_jumps, lam=lam)
            model.paths[n].generate_jump_distribution(dist=jump_dist, dist_params=jump_params)
                
        permeabilities[i] = model.calculate_permeability()

    df3 = pd.DataFrame()
    df3['lambda'] = lambdas
    df3['permeability'] = permeabilities
    df3['distribution'] = ['exponential']*len(lambdas)

    sns.lineplot(data=df1, x='lambda', y='permeability', color='tab:blue', label='equal')
    sns.lineplot(data=df2, x='lambda', y='permeability', color='tab:orange', label='normal')
    sns.lineplot(data=df3, x='lambda', y='permeability', color='tab:green', label='exponential')
    plt.xlabel('$\lambda$ (Angstrom)')
    plt.legend()
    plt.show()