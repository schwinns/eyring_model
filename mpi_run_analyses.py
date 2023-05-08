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
parallel_pores = False
compare_effective_barriers = False
estimate_dH_dS_barrier_distributions = False
estimate_dH_dS_jump_distributions = False
estimate_dH_dS_spread = False
compare_jump_lengths = False
plot_paths = False
test_path_convergence = True

# Inputs for testing barriers
T = 300
large = 18
large_barrier = large*R*T
small_barrier = large_barrier / 2
sigma = large_barrier / 3

if parallel_pores:

    n_paths = 50
    n_jumps = 50
    fill = True

    fig, ax = plt.subplots(3,1, figsize=(12,8), sharex=True)

    # ALL MEMBRANE BARRIERS EQUAL

    model_equal = EyringModel(T=T)
    dist = 'equal'
    params = {'mu' : large_barrier}

    # plot the membrane barrier distribution for each pore, overlapping
    effective_barriers = np.zeros(n_paths)
    for n in tqdm(range(n_paths)):
        model_equal.add_Path(dist=dist, dist_params=params, n_jumps=n_jumps)
        effective_barriers[n] = model_equal.paths[n].calculate_effective_barrier() / (R*T)
        sns.histplot(model_equal.paths[n].membrane_barriers / (R*T), edgecolor=None, ax=ax[0], stat='density', fill=fill, alpha=0.25)
        # model_equal.paths[n].plot_distribution(fill=fill, ax=ax[0])

    permeability = model_equal.calculate_permeability()
    effective_barrier_equal = model_equal.calculate_effective_barrier() / (R*T)

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
    params = {'mu' : large_barrier, 'sigma' : sigma}

    # plot the membrane barrier distribution for each pore, overlapping
    effective_barriers = np.zeros(n_paths)
    for n in tqdm(range(n_paths)):
        model_norm.add_Path(dist=dist, dist_params=params, n_jumps=n_jumps)
        effective_barriers[n] = model_norm.paths[n].calculate_effective_barrier() / (R*T)
        sns.histplot(model_norm.paths[n].membrane_barriers / (R*T), binwidth=1, edgecolor=None, ax=ax[1], stat='density', fill=fill, alpha=0.25)
        # model_norm.paths[n].plot_distribution(fill=fill, ax=ax[1])

    permeability = model_norm.calculate_permeability()
    effective_barrier_norm = model_norm.calculate_effective_barrier() / (R*T)

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
    params = {'beta' : large_barrier}

    # plot the membrane barrier distribution for each pore, overlapping
    effective_barriers = np.zeros(n_paths)
    for n in tqdm(range(n_paths)):
        model_exp.add_Path(dist=dist, dist_params=params, n_jumps=n_jumps)
        effective_barriers[n] = model_exp.paths[n].calculate_effective_barrier() / (R*T)
        sns.histplot(model_exp.paths[n].membrane_barriers / (R*T), binwidth=1, edgecolor=None, ax=ax[2], stat='density', fill=fill, alpha=0.25)
        # model_exp.paths[n].plot_distribution(fill=fill, ax=ax[2])

    permeability = model_exp.calculate_permeability()
    effective_barrier_exp = model_exp.calculate_effective_barrier() / (R*T)

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
    ax[2].set_xlim(-10,)

    # fig1, ax1 = plt.subplots(1,1, figsize=(6,6))
    # sns.barplot(data=df_norm, x='pores', y='permeability_percent', ax=ax1)
    # ax1.set_ylabel('percentage of permeability')
    # xmin, xmax = plt.xlim()
    # ymin, ymax = plt.ylim()
    # ax1.text(xmax*0.95, ymax*0.9, 'Max P: {:.4f}\nOverall P: {:.4f}'.format(df_norm['permeability'].max(), df_norm['permeability'].sum()), ha='right')

    # fig2, ax2 = plt.subplots(1,1, figsize=(6,6))
    # sns.barplot(data=df_exp, x='pores', y='permeability_percent', ax=ax2)
    # ax2.set_ylabel('percentage of permeability')
    # xmin, xmax = plt.xlim()
    # ymin, ymax = plt.ylim()
    # ax2.text(xmax*0.95, ymax*0.9, 'Max P: {:.4f}\nOverall P: {:.4f}'.format(df_exp['permeability'].max(), df_exp['permeability'].sum()), ha='right')
    
    # fig3, ax3 = plt.subplots(1,1, figsize=(6,6))
    # sns.lineplot(data=df_equal, x='pore_fraction', y='flux_fraction', ax=ax3, label='equal')
    # sns.lineplot(data=df_norm, x='pore_fraction', y='flux_fraction', ax=ax3, label='normal')
    # sns.lineplot(data=df_exp, x='pore_fraction', y='flux_fraction', ax=ax3, label='exponential')
    # ax3.set_xlabel('fraction of the pores')
    # ax3.set_ylabel('fraction of the flux')
    # # ax3.set_xlim(0,1)
    # # ax3.set_ylim(0,1)
    plt.show()


if compare_effective_barriers:

    dist = 'normal'
    params = {'mu' : large_barrier, 'sigma' : sigma}
    model = Path(T=T, dist=dist, dist_params=params)
    dG_eff = model.calculate_effective_barrier() / (R*T)
    model.membrane_barriers = model.membrane_barriers / (R*T)
    ax = model.plot_distribution(hist=True, color='tab:blue', binwidth=1, label='normal')
    ymin, ymax = plt.ylim()
    ax.axvline(dG_eff, ls='dashed', c='tab:blue')
    ax.text(dG_eff*1.1, ymax*0.9, '$\Delta G_{eff}$/RT')

    dist = 'exponential'
    params = {'beta' : large_barrier}
    model = Path(T=T, dist=dist, dist_params=params)
    dG_eff = model.calculate_effective_barrier() / (R*T)
    model.membrane_barriers = model.membrane_barriers / (R*T)
    model.plot_distribution(hist=True, color='tab:orange', binwidth=1, ax=ax, label='exponential')
    ymin, ymax = plt.ylim()
    ax.axvline(dG_eff, ls='dashed', c='tab:orange')
    ax.text(dG_eff*1.01, ymax*0.9, '$\Delta G_{eff}$/RT')
    
    ax.axvline(large, c='r')
    ax.text(large*0.5, ymax*0.9, 'mean', ha='left')
    ax.set_xlabel('$\Delta G_{M,j}$ / RT')
    plt.legend(loc='center')
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
            model.add_Path(dist=dist, dist_params={'mu' : large*R*300}) 

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
    deltas = np.zeros(len(lambdas))
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
        deltas[i] = np.array(model.deltas).mean()
        effective_barriers[i] = model.calculate_effective_barrier()

    df1 = pd.DataFrame()
    df1['lambda'] = lambdas
    df1['permeability'] = permeabilities
    df1['thickness'] = deltas
    df1['effective_barriers'] = effective_barriers*R*T
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
        deltas[i] = np.array(model.deltas).mean()
        effective_barriers[i] = model.calculate_effective_barrier()

    df2 = pd.DataFrame()
    df2['lambda'] = lambdas
    df2['permeability'] = permeabilities
    df2['thickness'] = deltas
    df2['effective_barriers'] = effective_barriers*R*T
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
        deltas[i] = np.array(model.deltas).mean()
        effective_barriers[i] = model.calculate_effective_barrier()

    df3 = pd.DataFrame()
    df3['lambda'] = lambdas
    df3['permeability'] = permeabilities
    df3['thickness'] = deltas
    df3['effective_barriers'] = effective_barriers*R*T
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
    plt.ylabel('$\Delta G_{eff}$/RT')
    plt.legend()
    plt.show()

if estimate_dH_dS_spread:

    n_paths = 1200
    temps = np.array([250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350]*3)

    dG_eff = np.zeros(len(temps))
    P = np.zeros(len(temps))
    X = np.zeros(len(temps))
    Y = np.zeros(len(temps))

    # MULTIVARIATE NORMAL

    print('\nMULTIVARIATE NORMAL')

    params = {
        'mu'  : np.array([4.5, 6]),
        'cov' : np.array([[0.001**2,0],
                          [0,0.001**2]])
    }

    dist = 'normal'
    multi = True

    for i, T in tqdm(enumerate(temps)):
        model = EyringModel(T=T)
        for n in range(n_paths):
            model.add_Path(n_jumps=200, lam=10)
            model.paths[n].generate_membrane_barriers(dist=dist, multi=multi, dist_params=params)

        P[i] = model.calculate_permeability()
        dG_eff[i] = model.calculate_effective_barrier()
        lam = model.get_lambda()
        delta = np.array(model.deltas).mean()
        X[i] = 1 / T
        Y[i] = np.log(P[i]*h*delta / (kB*T*lam**2))

    m, b = np.polyfit(X,Y,1)
    print(f'dH_eff : {-m*R}')
    print(f'dS_eff : {b*R} or -T dS_eff at 300 K: {-300*b*R}')
    print(f'dG_eff at 300 K: {dG_eff.mean()}')

    df1 = pd.DataFrame()
    df1['distribution'] = ['multi-variate normal']*len(temps)
    df1['temperature'] = temps
    df1['permeability'] = P
    df1['log permeability'] = np.log(P)
    df1['effective free energy'] = dG_eff
    df1['1/T'] = X
    df1['ln(P h del / kB T lam^2)'] = Y

    # # MULTIPLE EXPONENTIALS

    # print('\nMULTIPLE EXPONENTIALS:')

    # params = {
    #     'beta'  : np.array([4.25, 0.021]),
    # }

    # dist = 'exponential'
    # multi = True

    # for i, T in tqdm(enumerate(temps)):
    #     model = EyringModel(T=T)
    #     for n in range(n_paths):
    #         model.add_Path(n_jumps=200, lam=10)
    #         model.paths[n].generate_membrane_barriers(dist=dist, multi=multi, dist_params=params)

    #     dG_eff[i] = model.calculate_effective_barrier()
    #     P[i] = model.calculate_permeability()
    #     lam = model.get_lambda()
    #     delta = np.array(model.deltas).mean()
    #     X[i] = 1 / T
    #     Y[i] = np.log(P[i]*h*delta / (kB*T*lam**2))

    # m, b = np.polyfit(X,Y,1)
    # print(f'dH_eff : {-m*R}')
    # print(f'dS_eff : {b*R} or -T dS_eff at 300 K: {-300*b*R}')

    # df2 = pd.DataFrame()
    # df2['distribution'] = ['multiple exponentials']*len(temps)
    # df2['temperature'] = temps
    # df2['permeability'] = P
    # df2['effective free energy'] = dG_eff
    # df2['1/T'] = X
    # df2['ln(P h del / kB T lam^2)'] = Y

    # df = pd.concat((df1,df2))

    sns.lmplot(data=df1, x='1/T', y='ln(P h del / kB T lam^2)', hue='distribution', 
               scatter_kws={'alpha':0.75, 'edgecolor':'black'})

    plt.figure()
    sns.scatterplot(data=df1, x='temperature', y='permeability', hue='distribution')

    plt.figure()
    sns.scatterplot(data=df1, x='temperature', y='log permeability')

    plt.figure()
    sns.scatterplot(data=df1, x='temperature', y='effective free energy', hue='distribution')

    plt.show()

if test_path_convergence:

    # run this on all available processors
    from mpi4py import MPI

    def calculate_barriers_permeability(N):

        # Inputs for testing
        T = 300
        barrier = 10
        sigma = barrier / 3

        n_jumps = 50

        dist = 'exponential'
        params = {'beta' : barrier}

        model = EyringModel(T=T)
        for n in range(N):
            model.add_Path(dist=dist, dist_params=params, n_jumps=n_jumps, lam=10) # 10 Angstrom jump lengths over 50

        return model.calculate_effective_barrier(), model.calculate_permeability()

    ########### DISTRIBUTE THE WORK ACROSS PROCESSORS ##########

    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        fig1, ax1 = plt.subplots(1,1)
        fig2, ax2 = plt.subplots(1,1)
        
        # Increasing number of paths to see how effective barrier changes (when it stabilizes)
        n_paths = np.array([50,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,
                        1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,
                        3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000,4100,4200,4300,
                        4400,4500,4600,4700,4800,4900,5000], dtype='i')
        
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

    realizations = 100
    for r in range(realizations):

        if rank == 0:
            print(f'Realization {r+1}...')

        for i,N in enumerate(my_paths):
            effective_barriers[i], permeabilities[i] = calculate_barriers_permeability(N)

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
        ax1.plot(n_paths, avg_barrier / realizations, c='r')
        ax2.plot(n_paths, avg_permeability / realizations, c='r')

        ax1.set_xlabel('number of paths')
        ax1.set_ylabel('effective barrier')
        ax2.set_xlabel('number of paths')
        ax2.set_ylabel('permeability per path')
        plt.show()