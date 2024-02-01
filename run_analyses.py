# Script to run Eyring Model analyses

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from scipy.interpolate import CubicSpline
from scipy.stats import truncnorm
import statsmodels.api as sm

from eyring_model import EyringModel, Path

# Define global constants
global kB 
kB = 1.380649 * 10**-23    # Boltzmann (m^2 kg / s^2 K)
global h
h = 6.62607 * 10**-34      # Planck (m^2 kg / s)
global R
R = 1.9858775 * 10**-3     # universal gas (kcal / mol K)

def parallel_pores(dH_barrier, dS_barrier, dH_sigma, dS_sigma, dG_barrier, n_paths=2000, n_jumps=200, T=300, multi=True, output='figs/hist_effective_individual_barriers_no_penalty.pdf'):

    print(f'\nCalculating effective barriers and fractions of permeability for {n_paths} paths through the membrane...')

    fig, ax = plt.subplots(3,1, figsize=(7.25,3.55), sharex=True)

    # ALL MEMBRANE BARRIERS EQUAL

    model_equal = EyringModel(T=T)
    dist = 'equal'
    params = {'mu' : np.array([dH_barrier, dS_barrier])}

    print(f'\tfor equal barriers:')

    # plot the membrane barrier distribution for each pore, overlapping
    effective_barriers = np.zeros(n_paths)
    all_barriers = []
    for n in tqdm(range(n_paths)):
        model_equal.add_Path(n_jumps=n_jumps, area=model_equal.area/n_paths)
        model_equal.paths[n].generate_membrane_barriers(dist=dist, multi=multi, dist_params=params)
        effective_barriers[n] = model_equal.paths[n].calculate_effective_barrier()
        [all_barriers.append(b) for b in model_equal.paths[n].membrane_barriers]

    sns.histplot(all_barriers, edgecolor='black', ax=ax[0], stat='density', color='tab:gray', alpha=0.5, label='individual barriers', linewidth=0.5)
    permeability = model_equal.calculate_permeability()
    effective_barrier_equal = model_equal.calculate_effective_barrier()
    std_equal = np.std(all_barriers)
    mean_equal = np.mean(all_barriers)

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

    sns.histplot(effective_barriers, color='tab:gray', linewidth=1, ax=ax[0], 
                 stat='density', alpha=1, fill=False, label='path effective barriers')
    
    # NORMAL DISTRIBUTION OF BARRIERS

    model_norm = EyringModel(T=T)
    dist = 'normal'
    params = {'mu'  : np.array([dH_barrier, dS_barrier]),
              'cov' : np.array([[dH_sigma**2,0],
                                [0,dS_sigma**2]])}
    
    print(f'\tfor normal barriers:')

    # plot the membrane barrier distribution for each pore, overlapping
    effective_barriers = np.zeros(n_paths)
    all_barriers = []
    for n in tqdm(range(n_paths)):
        model_norm.add_Path(n_jumps=n_jumps, area=model_norm.area/n_paths)
        model_norm.paths[n].generate_membrane_barriers(dist=dist, multi=multi, dist_params=params)
        effective_barriers[n] = model_norm.paths[n].calculate_effective_barrier()
        [all_barriers.append(b) for b in model_norm.paths[n].membrane_barriers]

    sns.histplot(all_barriers, binwidth=1, edgecolor='black', ax=ax[1], stat='density', color='tab:blue', alpha=0.5, label='individual barriers', linewidth=0.5)
    permeability = model_norm.calculate_permeability()
    effective_barrier_norm = model_norm.calculate_effective_barrier()
    std_norm = np.std(all_barriers)
    mean_norm = np.mean(all_barriers)

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

    sns.histplot(effective_barriers, binwidth=1, color='tab:blue', linewidth=1, ax=ax[1],
                  stat='density', alpha=1, fill=False, label='path effective barriers')
    
    # EXPONENTIAL DISTRIBUTION OF BARRIERS

    model_exp = EyringModel(T=T)
    dist = 'exponential'
    params = {'beta' : np.array([dH_barrier, dS_barrier])}

    print(f'\tfor exponential barriers:')

    # plot the membrane barrier distribution for each pore, overlapping
    effective_barriers = np.zeros(n_paths)
    all_barriers = []
    for n in tqdm(range(n_paths)):
        model_exp.add_Path(n_jumps=n_jumps, area=model_exp.area/n_paths)
        model_exp.paths[n].generate_membrane_barriers(dist=dist, multi=multi, dist_params=params)
        effective_barriers[n] = model_exp.paths[n].calculate_effective_barrier()
        [all_barriers.append(b) for b in model_exp.paths[n].membrane_barriers]

    sns.histplot(all_barriers, binwidth=1, edgecolor='black', ax=ax[2], stat='density', color='tab:orange', alpha=0.5, label='individual barriers', linewidth=0.5)
    permeability = model_exp.calculate_permeability()
    effective_barrier_exp = model_exp.calculate_effective_barrier()
    std_exp = np.std(all_barriers)
    mean_exp = np.mean(all_barriers)

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

    sns.histplot(effective_barriers, binwidth=1, color='tab:orange', linewidth=1, ax=ax[2],
                  stat='density', alpha=1, fill=False, label='path effective barriers')
    
    # PLOTTING

    # plot the effective barrier
    ax[0].axvline(effective_barrier_equal, ls='dashed', c='k', label='$\Delta G_{eff}^{\ddag}$', lw=1)
    ax[0].legend(frameon=False)
    ax[0].set_ylabel('Density')
    # ax[0].set_title('Series of equal barriers, standard deviation = 0 kcal/mol')

    ax[1].axvline(effective_barrier_norm, ls='dashed', c='k', label='$\Delta G_{eff}^{\ddag}$', lw=1)
    ax[1].legend(frameon=False)
    ax[1].set_ylabel('Density')
    # ax[1].set_title('$\Delta H^{\ddag}_{M,i,j}$, $\Delta S^{\ddag}_{M,i,j}$ normally distributed, standard deviation = 3 kcal/mol')

    ax[2].axvline(effective_barrier_exp, ls='dashed', c='k', label='$\Delta G_{eff}^{\ddag}$', lw=1)
    ax[2].legend(frameon=False)
    ax[2].set_ylabel('Density')
    # ax[2].set_title('$\Delta H^{\ddag}_{M,i,j}$, $\Delta S^{\ddag}_{M,i,j}$ exponentially distributed, standard deviation = 10 kcal/mol')

    ax[2].set_xlabel('$\Delta G_{M,i,j}^{\ddag}$ (kcal/mol)')
    ax[2].set_xlim(0,100)

    plt.savefig(output)

    print(f'Means: {mean_equal} (equal), {mean_norm} (normal), {mean_exp} (exponential)')
    print(f'Standard deviations: {std_equal} (equal), {std_norm} (normal), {std_exp} (exponential)')
    print(f'Effective barriers: {effective_barrier_equal} (equal) {effective_barrier_norm} (normal), {effective_barrier_exp} (exponential)')
    
    # fig3, ax3 = plt.subplots(1,1, figsize=(6,6))
    # sns.lineplot(data=df_equal, x='pore_fraction', y='flux_fraction', ax=ax3, label='equal')
    # sns.lineplot(data=df_norm, x='pore_fraction', y='flux_fraction', ax=ax3, label='normal')
    # sns.lineplot(data=df_exp, x='pore_fraction', y='flux_fraction', ax=ax3, label='exponential')
    # ax3.set_xlabel('fraction of the pores')
    # ax3.set_ylabel('fraction of the flux')

    plt.show()

    return model_equal, model_norm, model_exp


def compare_effective_barriers(dH_barrier, dS_barrier, dH_sigma, dS_sigma, dG_barrier, T=300, multi=True):

    print(f'\nComparing effective barriers for a single path through the membrane...')

    fig, ax = plt.subplots(1,1, figsize=(7.25,1.95))

    # Generate normally distributed barriers
    dist = 'normal'
    params = {'mu'  : np.array([dH_barrier, dS_barrier]),
              'cov' : np.array([[dH_sigma**2,0],
                                [0,dS_sigma**2]])}

    model = Path(T=T, n_jumps=200)
    model.generate_membrane_barriers(dist=dist, multi=multi, dist_params=params)
    dG_eff = model.calculate_effective_barrier()
    sns.histplot(model.membrane_barriers, color='tab:blue', stat='probability', edgecolor='k', binwidth=1, bins=50, alpha=0.5, ax=ax, label='normal barriers', linewidth=0.5)
    ymin, ymax = plt.ylim()
    ax.axvline(dG_eff, ls='dashed', c='tab:blue')
    ax.text(dG_eff*1.05, ymax*0.9, '$\Delta G_{eff}^{\ddag}$', color='tab:blue')

    # Generate exponentially distributed barriers
    dist = 'exponential'
    params = {'beta'  : np.array([dH_barrier, dS_barrier])}

    model = Path(T=T)
    model.generate_membrane_barriers(dist=dist, multi=multi, dist_params=params)
    dG_eff = model.calculate_effective_barrier()
    sns.histplot(model.membrane_barriers, color='tab:orange', stat='probability', edgecolor='k', binwidth=1, bins=50, alpha=0.5, ax=ax, label='exponential barriers', linewidth=0.5)
    ymin, ymax = plt.ylim()
    ax.axvline(dG_eff, ls='dashed', c='tab:orange')
    ax.text(dG_eff*1.01, ymax*0.9, '$\Delta G_{eff}^{\ddag}$', color='tab:orange')
    
    # formatting
    ax.set_xlabel('$\Delta G_{M,j}^{\ddag}$ (kcal/mol)')
    ax.set_ylabel('Density')
    ax.set_xlim(0,)
    plt.legend(loc='center', frameon=False, ncol=1)
    fig.savefig('figs/effective_barrier_distribution_comparison.pdf')
    plt.show()

def plot_paths(n, dH_barrier, dS_barrier, dH_sigma, dS_sigma, T=300, multi=True):

    print(f'\nPlotting {n} realizations of barrier paths through the membrane...')

    # fig, ax = plt.subplots(n,1, figsize=(12.8,n*2+3.5), sharex=True, sharey=True)
    fig, ax = plt.subplots(n,1, figsize=(7.25,1.95), sharex=True, sharey=True)

    for i in range(n):
    
        # instantiate the model and generate barriers
        dist = 'normal'
        params = {
                'mu'  : np.array([dH_barrier, dS_barrier]),
                'cov' : np.array([[dH_sigma**2,0],
                                  [0,dS_sigma**2]])
                }
        model = Path(T=T, n_jumps=100)
        model.generate_membrane_barriers(dist=dist, multi=multi, dist_params=params)
        dG_eff = model.calculate_effective_barrier()

        # get jumps and barriers for plotting
        jumps = model.jump_lengths.cumsum()
        barriers = model.membrane_barriers

        # create a cubic spline so the barrier profile is smooth
        path_spline = CubicSpline(jumps, barriers, bc_type='natural')
        xs = np.linspace(0, jumps.max(), num=2000)
        ys = path_spline(xs)

        # plot the path
        if n > 1:
            ax[i].plot(xs, ys, color='tab:blue', alpha=1, label='normal barriers')
            ax[i].axhline(dG_eff, c='tab:blue', ls='dashed')
            ax[i].text(model.lam, dG_eff+2, '$\Delta G_{eff}^{\ddag}$', ha='right', color='tab:blue')
        else:
            ax.plot(xs, ys, color='tab:blue', alpha=1, label='normal barriers')
            ax.axhline(dG_eff, c='tab:blue', ls='dashed')
            ax.text(model.lam, dG_eff+2, '$\Delta G_{eff}^{\ddag}$', ha='right', color='tab:blue')

        # repeat for exponentially distributed barriers
        dist = 'exponential'
        params = {'beta'  : np.array([dH_barrier, dS_barrier])}
        model = Path(T=T, n_jumps=100)
        model.generate_membrane_barriers(dist=dist, multi=multi, dist_params=params)
        dG_eff = model.calculate_effective_barrier()

        jumps = model.jump_lengths.cumsum()
        barriers = model.membrane_barriers

        path_spline = CubicSpline(jumps, barriers, bc_type='natural')
        xs = np.linspace(0, jumps.max(), num=2000)
        ys = path_spline(xs)

        if n > 1:
            ax[i].plot(xs, ys, color='tab:orange', alpha=0.75, label='exponential barriers')
            ax[i].axhline(dG_eff, c='tab:orange', ls='dashed')
            ax[i].text(model.lam, dG_eff+2, '$\Delta G_{eff}^{\ddag}$', ha='right', color='tab:orange')

            ax[i].set_ylabel('$\Delta G_{M,j}^{\ddag}$ (kcal/mol)')
            ax[i].legend(loc='upper right', ncol=2,
                        frameon=False, borderpad=0.2)
            ax[i].set_ylim(0,dG_eff*1.2)
        else:
            ax.plot(xs, ys, color='tab:orange', alpha=0.75, label='exponential barriers')
            ax.axhline(dG_eff, c='tab:orange', ls='dashed')
            ax.text(model.lam, dG_eff+2, '$\Delta G_{eff}^{\ddag}$', ha='right', color='tab:orange')

            ax.set_ylabel('$\Delta G_{M,j}^{\ddag}$ (kcal/mol)')
            ax.legend(loc='upper right', ncol=2,
                        frameon=False, borderpad=0.2)
            ax.set_ylim(0,dG_eff*1.2)

    # some overall formatting
    if n > 1:
        ax[i].set_xlim(-9*model.lam,)
        ax[i].set_xlabel('Transport Coordinate ($\mathrm{\AA}$)')
    else:
        ax.set_xlim(-9*model.lam,)
        ax.set_xlabel('Transport Coordinate ($\mathrm{\AA}$)')
    
    plt.savefig('figs/barrier_profile_1_path.pdf')
    plt.show()

def compare_jump_lengths(dH_barrier, dS_barrier, n_paths, delta=400, T=300, multi=True):

    dist = 'equal'
    params = params = {'mu'  : np.array([dH_barrier, dS_barrier])}

    lambdas = [1,2,3,4,5,6,7,8,9,10] # list of jump lengths to compare
    n_replicates = 10

    fig, ax = plt.subplots(1,1, figsize=(6,6))
    # fig, ax1 = plt.subplots(3,1, figsize=(6,6), sharex=True)

    print(f'\nComparing effective barriers for distributions of jump lengths with mean overall thickness {delta} Angstroms...')

    # Jump lengths EQUAL

    print(f'\tfor equal dsitribution:')

    jump_dist = 'equal'
    effective_barriers = np.zeros((len(lambdas),2))

    for i,lam in tqdm(enumerate(lambdas)):

        deltas = []
        lam_barriers = np.zeros(n_replicates)
        for r in range(n_replicates):

            model = EyringModel(T=T)
            n_jumps_mu = delta / lam
            n_jumps_sig = 3

            # add all parallel paths
            for n in range(n_paths):
                jump_params = {'mu' : lam}
                n_jumps = int(np.random.default_rng().normal(loc=n_jumps_mu, scale=n_jumps_sig))
                model.add_Path(n_jumps=n_jumps, lam=lam)
                model.paths[n].generate_membrane_barriers(dist=dist, multi=multi, dist_params=params)
                model.paths[n].generate_jump_distribution(dist=jump_dist, dist_params=jump_params)

            [deltas.append(d) for d in model.deltas]
            lam_barriers[r] = model.calculate_effective_barrier()
    
        effective_barriers[i,0] = lam_barriers.mean()
        effective_barriers[i,1] = lam_barriers.std()
        # sns.histplot(deltas, edgecolor='black', ax=ax1[0], stat='density', color='tab:gray', alpha=0.75)

    ax.errorbar(lambdas, effective_barriers[:,0], yerr=effective_barriers[:,1], c='tab:gray', label='equal', fmt='o')
    # ax.plot(lambdas, effective_barriers[:,0], c='tab:gray', label='equal')
    # ax.fill_between(lambdas, effective_barriers[:,0]-effective_barriers[:,1], effective_barriers[:,0]+effective_barriers[:,1], alpha=0.25, color='tab:gray')
    print(f'Effective barrier changes from {effective_barriers[0,0]:.4f} +/- {effective_barriers[0,1]:.4f} to {effective_barriers[-1,0]:.4f} +/- {effective_barriers[-1,1]:.4f} as mean jump length increases from {lambdas[0]} to {lambdas[-1]}')

    # Jump lengths NORMAL

    print(f'\tfor normal distribution:')
    
    jump_dist = 'normal'

    for i,lam in tqdm(enumerate(lambdas)):

        deltas = []
        lam_barriers = np.zeros(n_replicates)
        for r in range(n_replicates):

            model = EyringModel(T=T)
            n_jumps_mu = delta / lam
            n_jumps_sig = 3

            # add all parallel paths
            for n in range(n_paths):
                jump_params = {'mu' : lam, 'sigma' : lam/4}
                n_jumps = int(np.random.default_rng().normal(loc=n_jumps_mu, scale=n_jumps_sig))
                model.add_Path(n_jumps=n_jumps, lam=lam)
                model.paths[n].generate_membrane_barriers(dist=dist, multi=multi, dist_params=params)
                model.paths[n].generate_jump_distribution(dist=jump_dist, dist_params=jump_params)

            [deltas.append(d) for d in model.deltas]
            lam_barriers[r] = model.calculate_effective_barrier()
    
        effective_barriers[i,0] = lam_barriers.mean()
        effective_barriers[i,1] = lam_barriers.std()
        # sns.histplot(deltas, edgecolor='black', ax=ax1[1], stat='density', color='tab:blue', alpha=0.75)
    
    ax.errorbar(lambdas, effective_barriers[:,0], yerr=effective_barriers[:,1], c='tab:blue', label='normal', fmt='o')
    # ax.plot(lambdas, effective_barriers[:,0], c='tab:blue', label='normal')
    # ax.fill_between(lambdas, effective_barriers[:,0]-effective_barriers[:,1], effective_barriers[:,0]+effective_barriers[:,1], alpha=0.25, color='tab:blue')
    print(f'Effective barrier changes from {effective_barriers[0,0]:.4f} +/- {effective_barriers[0,1]:.4f} to {effective_barriers[-1,0]:.4f} +/- {effective_barriers[-1,1]:.4f} as mean jump length increases from {lambdas[0]} to {lambdas[-1]}')

    # Jump lengths EXPONENTIAL
    
    jump_dist = 'exponential'

    print(f'\tfor exponential distribution:')

    for i,lam in tqdm(enumerate(lambdas)):

        deltas = []
        lam_barriers = np.zeros(n_replicates)
        for r in range(n_replicates):

            model = EyringModel(T=T)
            n_jumps_mu = delta / lam
            n_jumps_sig = 3

            # add all parallel paths
            for n in range(n_paths):
                jump_params = {'beta' : lam}
                n_jumps = int(np.random.default_rng().normal(loc=n_jumps_mu, scale=n_jumps_sig))
                model.add_Path(n_jumps=n_jumps, lam=lam)
                model.paths[n].generate_membrane_barriers(dist=dist, multi=multi, dist_params=params)
                model.paths[n].generate_jump_distribution(dist=jump_dist, dist_params=jump_params)

            [deltas.append(d) for d in model.deltas]
            lam_barriers[r] = model.calculate_effective_barrier()
    
        effective_barriers[i,0] = lam_barriers.mean()
        effective_barriers[i,1] = lam_barriers.std()
        # sns.histplot(deltas, edgecolor='black', ax=ax1[2], stat='density', color='tab:orange', alpha=0.75)
    
    ax.errorbar(lambdas, effective_barriers[:,0], yerr=effective_barriers[:,1], c='tab:orange', label='exponential', fmt='o')
    # ax.plot(lambdas, effective_barriers[:,0], c='tab:orange', label='exponential')
    # ax.fill_between(lambdas, effective_barriers[:,0]-effective_barriers[:,1], effective_barriers[:,0]+effective_barriers[:,1], alpha=0.25, color='tab:orange')
    print(f'Effective barrier changes from {effective_barriers[0,0]:.4f} +/- {effective_barriers[0,1]:.4f} to {effective_barriers[-1,0]:.4f} +/- {effective_barriers[-1,1]:.4f} as mean jump length increases from {lambdas[0]} to {lambdas[-1]}')
    
    ax.set_xlabel('Mean jumpth length ($\mathrm{\AA}$)')
    ax.set_ylabel('$\Delta G_{eff}^{\ddag}$ (kcal/mol)')
    ax.set_xticks(np.arange(11))
    ax.set_xlim(0,10)
    ax.legend(fontsize=16)

    # ax1[2].set_xlabel('membrane thickness ($\r{A}$)')

    plt.savefig('figs/jump_length_effects.png')
    plt.show()

def estimate_dH_dS(dH_barrier, dS_barrier, dH_sigma, dS_sigma, n_paths, area=1e7, plot=False):

    print(f'\nEstimating the effective enthalpic and entropic barriers...')

    multi = True

    temps = np.array([250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350])

    dG_eff = np.zeros(len(temps))
    P = np.zeros(len(temps))
    X = np.zeros(len(temps))
    Y = np.zeros(len(temps))
    max_barriers_norm = np.zeros((len(temps),n_paths,2))
    max_barriers_exp = np.zeros((len(temps),n_paths,2))

    hist_alpha = 0.5
    error_alpha = 0.1

    fig, ax = plt.subplots(2,3, figsize=(7.25,5.5), sharex=False, sharey=True)

    # MULTIVARIATE NORMAL

    print('\nNORMALLY DISTRIBUTED:')

    params = {
        'mu'  : np.array([dH_barrier, dS_barrier]),
        'cov' : np.array([[dH_sigma**2,0],
                          [0,dS_sigma**2]])
    }

    dist = 'normal'

    all_dH = []
    all_dS = []
    all_dG = []

    for i, T in tqdm(enumerate(temps)):
        model = EyringModel(T=T, A=area)
        for n in range(n_paths):
            model.add_Path(n_jumps=200, lam=10)
            model.paths[n].generate_membrane_barriers(dist=dist, multi=multi, dist_params=params)
            max_barriers_norm[i,n,:] = np.array([model.paths[n].enthalpic_barriers.max(), model.paths[n].entropic_barriers.min()])
            if plot:
                [all_dH.append(b) for b in model.paths[n].enthalpic_barriers]
                [all_dS.append(-T*b) for b in model.paths[n].entropic_barriers]
                if T == 300:
                    [all_dG.append(b) for b in model.paths[n].membrane_barriers]

        P[i] = model.calculate_permeability() / 60 / 60 / 1000 * 10**9 * 10
        dG_eff[i] = model.calculate_effective_barrier()
        lam = model.get_lambda()
        delta = np.array(model.deltas).mean()
        X[i] = 1 / T
        Y[i] = np.log(P[i]*h*delta / (kB*T*lam**2))

    sns.histplot(all_dH, ax=ax[0,0], stat='probability', alpha=hist_alpha, facecolor='tab:blue', edgecolor=None)
    sns.histplot(all_dS, ax=ax[0,1], stat='probability', alpha=hist_alpha, facecolor='tab:blue', edgecolor=None)
    sns.histplot(all_dG, ax=ax[0,2], stat='probability', alpha=hist_alpha, facecolor='tab:blue', edgecolor=None)

    dHm = model.paths[n].enthalpic_barriers.mean()
    dSm = model.paths[n].entropic_barriers.mean()
    dGm = model.paths[n].membrane_barriers.mean()
    print(f'\nSingle path dH: {dHm}')
    print(f'Single path dS: {dSm} or -T dS at {T} K: {-T*dSm}')
    print(f'Single path dG: {dGm}')
    print(f'Many path contribution R ln(sum(A_i/A)): {R*np.log(np.sum(model.areas) / model.area)} or -RT ln(sum(A_i/A)) at 300 K: {-R*300*np.log(np.sum(model.areas) / model.area)}')

    avg_dH = np.mean(all_dH)
    avg_dS = np.mean(all_dS)
    avg_dG = np.mean(all_dG)
    sem_dH = np.std(all_dH) / np.sqrt(np.size(all_dH))
    sem_dS = np.std(all_dS) / np.sqrt(np.size(all_dS))
    sem_dG = np.std(all_dG) / np.sqrt(np.size(all_dG))
    print(f'\nAverage dH: {avg_dH} +/- {sem_dH}')
    print(f'Average dS: {avg_dS} +/- {sem_dS}')
    print(f'Average dG: {avg_dG} +/- {sem_dG}')

    A = sm.add_constant(X)
    ols = sm.OLS(Y, A)
    results = ols.fit()
    b, m = results.params
    be, me = results.bse

    eff_dH = np.array([-m*R, me*R]) # estimate, error
    eff_dS = -300*np.array([b*R, be*R])
    # eff_dG = np.array([eff_dH[0]-300*eff_dS[0], np.sqrt(eff_dH[1]**2 + (300*eff_dS[1])**2)])
    eff_dG = np.array([eff_dH[0]+eff_dS[0], np.sqrt(eff_dH[1]**2 + (eff_dS[1])**2)])

    print(f'\ndH_eff : {eff_dH[0]} +/- {eff_dH[1]}')
    print(f'dS_eff : {eff_dS[0]} +/- {eff_dS[1]} or -T dS_eff at 300 K: {-300*eff_dS[0]} +/- {300*eff_dS[1]}')
    print(f'dG_eff at 300 K from averaged effective barriers: {dG_eff.mean()} or from dH_eff and dS_eff: {eff_dG[0]} +/- {eff_dG[1]}')

    A = sm.add_constant(X)
    ols = sm.OLS(np.log(P), A)
    results = ols.fit()
    b, m = results.params
    be, me = results.bse
    print(f'\nArrhenius barrier to permeability: {-m*R} kcal/mol')
    print(f'Arrhenius barrier calculated from enthalpic barrier: {eff_dH[0]} kcal/mol')
    print(f'Arrhenius prefactor: {np.exp(b)} A/s')
    print(f'Arrhenius prefactor calculated from entropic barrier: {lam**2/delta * kB*300/h * np.exp(eff_dS[0]/R)} A/s')

    res = np.hstack((eff_dH, eff_dS, eff_dG))

    if plot:
        # plot effective, single path, mean barriers
        ax[0,0].set_title('Normally distributed $\Delta H_{M,i,j}^{\ddag}$', fontsize=8)
        ax[0,1].set_title('Normally distributed $-T \Delta S_{M,i,j}^{\ddag}$', fontsize=8)
        # ax[0,2].set_title('$\Delta G_{M,i,j}^{\ddag}$ at 300 K from normal $\Delta H_{M,i,j}^{\ddag}$ and $\Delta S_{M,i,j}^{\ddag}$', fontsize=8)
        ax[0,2].set_title('$\Delta G_{M,i,j}^{\ddag}$ at 300 K', fontsize=8)

        ax[0,0].axvline(eff_dH[0], ls='dashed', c='k', label='$\Delta H_{eff}^{\ddag}$', lw=1)
        ax[0,0].axvspan(eff_dH[0] - eff_dH[1], eff_dH[0] + eff_dH[1], facecolor='k', edgecolor=None, alpha=error_alpha)
        ax[0,0].axvline(avg_dH, ls='dashed', c='red', label='mean', lw=1)
        ax[0,0].axvspan(avg_dH - sem_dH, avg_dH + sem_dH, facecolor='red', edgecolor=None, alpha=error_alpha)
        
        ax[0,1].axvline(eff_dS[0], ls='dashed', c='k', label='$-T \Delta S_{eff}^{\ddag}$', lw=1)
        ax[0,1].axvspan(eff_dS[0] - eff_dS[1], eff_dS[0] + eff_dS[1], facecolor='k', edgecolor=None, alpha=error_alpha)
        ax[0,1].axvline(avg_dS, ls='dashed', c='red', label='mean', lw=1)
        ax[0,1].axvspan(avg_dS - sem_dS, avg_dS + sem_dS, facecolor='red', edgecolor=None, alpha=error_alpha)

        ax[0,2].axvline(eff_dG[0], ls='dashed', c='k', label='$\Delta G_{eff}^{\ddag}$', lw=1)
        ax[0,2].axvspan(eff_dG[0] - eff_dG[1], eff_dG[0] + eff_dG[1], facecolor='k', edgecolor=None, alpha=error_alpha)
        ax[0,2].axvline(avg_dG, ls='dashed', c='red', label='mean', lw=1)
        ax[0,2].axvspan(avg_dG - sem_dG, avg_dG + sem_dG, facecolor='red', edgecolor=None, alpha=error_alpha)

        ax[0,0].set_ylabel('Density')
        ax[0,1].set_ylabel(None)
        ax[0,2].set_ylabel(None)

        ax[0,0].set_xlim(0,)
        ax[0,1].set_xlim(0,)
        ax[0,2].set_xlim(0,)

        # ax[0,0].tick_params('y', labelrotation=45)
        # ax[0,1].tick_params('y', labelrotation=45)
        # ax[0,2].tick_params('y', labelrotation=45)

        ax[0,0].legend(frameon=False, ncol=1)
        ax[0,1].legend(frameon=False, ncol=1)
        ax[0,2].legend(frameon=False, ncol=1)

    df1 = pd.DataFrame()
    df1['distribution'] = ['multi-variate normal']*len(temps)
    df1['temperature'] = temps
    df1['permeability'] = P
    df1['effective free energy'] = dG_eff
    df1['1/T'] = X
    df1['ln(P/T)'] = Y

    fig1, ax1 = plt.subplots(1,1, figsize=(8,8))
    ax1 = sns.regplot(x='1/T', y='ln(P/T)', data=df1, ax=ax1)

    # MULTIPLE EXPONENTIALS

    print('\nEXPONENTIALLY DISTRIBUTED:')

    params = {'beta'  : np.array([dH_barrier, dS_barrier])}

    dist = 'exponential'

    all_dH = []
    all_dS = []
    all_dG = []

    for i, T in tqdm(enumerate(temps)):
        model = EyringModel(T=T, A=area)
        for n in range(n_paths):
            model.add_Path(n_jumps=200, lam=10)
            model.paths[n].generate_membrane_barriers(dist=dist, multi=multi, dist_params=params)
            max_barriers_exp[i,n,:] = np.array([model.paths[n].enthalpic_barriers.max(), model.paths[n].entropic_barriers.min()])
            if plot:
                [all_dH.append(b) for b in model.paths[n].enthalpic_barriers]
                [all_dS.append(-T*b) for b in model.paths[n].entropic_barriers]
                if T == 300:
                    [all_dG.append(b) for b in model.paths[n].membrane_barriers]

        dG_eff[i] = model.calculate_effective_barrier()
        P[i] = model.calculate_permeability() / 60 / 60 / 1000 * 10**9 * 10
        lam = model.get_lambda()
        delta = np.array(model.deltas).mean()
        X[i] = 1 / T
        Y[i] = np.log(P[i]*h*delta / (kB*T*lam**2))

    sns.histplot(all_dH, ax=ax[1,0], stat='probability', alpha=hist_alpha, facecolor='tab:orange', edgecolor=None)
    sns.histplot(all_dS, ax=ax[1,1], stat='probability', alpha=hist_alpha, facecolor='tab:orange', edgecolor=None)
    sns.histplot(all_dG, ax=ax[1,2], stat='probability', alpha=hist_alpha, facecolor='tab:orange', edgecolor=None)

    dHm = model.paths[n].enthalpic_barriers.mean()
    dSm = model.paths[n].entropic_barriers.mean()
    dGm = model.paths[n].membrane_barriers.mean()
    print(f'\nSingle path dH: {dHm}')
    print(f'Single path dS: {dSm} or -TdS at {T} K: {-T*dSm}')
    print(f'Single path dG: {dGm}')
    print(f'Many path contribution R ln(sum(A_i/A)): {R*np.log(np.sum(model.areas) / model.area)} or -RT ln(sum(A_i/A)) at 300 K: {-R*300*np.log(np.sum(model.areas) / model.area)}')

    avg_dH = np.mean(all_dH)
    avg_dS = np.mean(all_dS)
    avg_dG = np.mean(all_dG)
    sem_dH = np.std(all_dH) / np.sqrt(np.size(all_dH))
    sem_dS = np.std(all_dS) / np.sqrt(np.size(all_dS))
    sem_dG = np.std(all_dG) / np.sqrt(np.size(all_dG))
    print(f'\nAverage dH: {avg_dH} +/- {sem_dH}')
    print(f'Average dS: {avg_dS} +/- {sem_dS}')
    print(f'Average dG: {avg_dG} +/- {sem_dG}')

    A = sm.add_constant(X)
    ols = sm.OLS(Y, A)
    results = ols.fit()
    b, m = results.params
    be, me = results.bse
    
    eff_dH = np.array([-m*R, me*R]) # estimate, error
    eff_dS = -300*np.array([b*R, be*R])
    # eff_dG = np.array([eff_dH[0]-300*eff_dS[0], np.sqrt(eff_dH[1]**2 + (300*eff_dS[1])**2)])
    eff_dG = np.array([eff_dH[0]+eff_dS[0], np.sqrt(eff_dH[1]**2 + (eff_dS[1])**2)])

    print(f'\ndH_eff : {eff_dH[0]} +/- {eff_dH[1]}')
    print(f'dS_eff : {eff_dS[0]} +/- {eff_dS[1]} or -T dS_eff at 300 K: {-300*eff_dS[0]} +/- {300*eff_dS[1]}')
    print(f'dG_eff at 300 K from averaged effective barriers: {dG_eff.mean()} or from dH_eff and dS_eff: {eff_dG[0]} +/- {eff_dG[1]}')

    A = sm.add_constant(X)
    ols = sm.OLS(np.log(P), A)
    results = ols.fit()
    b, m = results.params
    be, me = results.bse
    print(f'\nArrhenius barrier to permeability: {-m*R} kcal/mol')
    print(f'Arrhenius barrier calculated from enthalpic barrier: {eff_dH[0]} kcal/mol')
    print(f'Arrhenius prefactor: {np.exp(b)} A/s')
    print(f'Arrhenius prefactor calculated from entropic barrier: {lam**2/delta * kB*300/h * np.exp(eff_dS[0]/R)} A/s')

    df2 = pd.DataFrame()
    df2['distribution'] = ['multiple exponentials']*len(temps)
    df2['temperature'] = temps
    df2['permeability'] = P
    df2['effective free energy'] = dG_eff
    df2['1/T'] = X
    df2['ln(P/T)'] = Y

    fig2, ax2 = plt.subplots(1,1, figsize=(8,8))
    ax2 = sns.regplot(x='1/T', y='ln(P/T)', data=df2, ax=ax2)

    if plot:
        # plot effective, single path, mean barriers
        ax[1,0].set_title('Exponentially distributed $\Delta H_{M,i,j}^{\ddag}$', fontsize=8)
        ax[1,1].set_title('Exponentially distributed $-T \Delta S_{M,i,j}^{\ddag}$', fontsize=8)
        # ax[1,2].set_title('$\Delta G_{M,i,j}^{\ddag}$ at 300 K from exponential $\Delta H_{M,i,j}^{\ddag}$ and $\Delta S_{M,i,j}^{\ddag}$', fontsize=8)
        ax[1,2].set_title('$\Delta G_{M,i,j}^{\ddag}$ at 300 K', fontsize=8)

        ax[1,0].axvline(eff_dH[0], ls='dashed', c='k', label='$\Delta H_{eff}^{\ddag}$', lw=1)
        ax[1,0].axvspan(eff_dH[0] - eff_dH[1], eff_dH[0] + eff_dH[1], facecolor='k', edgecolor=None, alpha=error_alpha)
        ax[1,0].axvline(avg_dH, ls='dashed', c='red', label='mean', lw=1)
        ax[1,0].axvspan(avg_dH - sem_dH, avg_dH + sem_dH, facecolor='red', edgecolor=None, alpha=error_alpha)
        
        ax[1,1].axvline(eff_dS[0], ls='dashed', c='k', label='$-T \Delta S_{eff}^{\ddag}$', lw=1)
        ax[1,1].axvspan(eff_dS[0] - eff_dS[1], eff_dS[0] + eff_dS[1], facecolor='k', edgecolor=None, alpha=error_alpha)
        ax[1,1].axvline(avg_dS, ls='dashed', c='red', label='mean', lw=1)
        ax[1,1].axvspan(avg_dS - sem_dS, avg_dS + sem_dS, facecolor='red', edgecolor=None, alpha=error_alpha)

        ax[1,2].axvline(eff_dG[0], ls='dashed', c='k', label='$\Delta G_{eff}^{\ddag}$', lw=1)
        ax[1,2].axvspan(eff_dG[0] - eff_dG[1], eff_dG[0] + eff_dG[1], facecolor='k', edgecolor=None, alpha=error_alpha)
        ax[1,2].axvline(avg_dG, ls='dashed', c='red', label='mean', lw=1)
        ax[1,2].axvspan(avg_dG - sem_dG, avg_dG + sem_dG, facecolor='red', edgecolor=None, alpha=error_alpha)

        ax[1,0].set_xlim(0,25)
        ax[1,1].set_xlim(0,60)
        ax[1,2].set_xlim(0,60)

        ax[1,0].set_xlabel('$\Delta H_{M,i,j}^{\ddag}$ (kcal/mol)')
        ax[1,1].set_xlabel('$-T \Delta S_{M,i,j}^{\ddag}$ (kcal/mol)')
        ax[1,2].set_xlabel('$\Delta G_{M,i,j}^{\ddag}$ (kcal/mol)')
        ax[1,0].set_ylabel('Density')
        ax[1,1].set_ylabel(None)
        ax[1,2].set_ylabel(None)

        # ax[1,0].tick_params('y', labelrotation=45)
        # ax[1,1].tick_params('y', labelrotation=45)
        # ax[1,2].tick_params('y', labelrotation=45)

        ax[1,0].legend(frameon=False, ncol=1)
        ax[1,1].legend(frameon=False, ncol=1)
        ax[1,2].legend(frameon=False, ncol=1)

    if plot:
        # plot maximum barriers
        fig3, ax3 = plt.subplots(2,2, figsize=(14,10), sharex=True)

        paths = np.arange(1, n_paths+1)

        for i, T in enumerate(temps):

            # plot the max enthalpies for a given temperature
            ax3[0,0].scatter(paths, max_barriers_norm[i,:,0], s=2, alpha=0.1, facecolor='tab:blue')
            ax3[1,0].scatter(paths, max_barriers_exp[i,:,0], s=2, alpha=0.1, facecolor='tab:orange')

            # plot the max entropies for a given temperature
            ax3[0,1].scatter(paths, max_barriers_norm[i,:,1], s=2, alpha=0.1, facecolor='tab:blue')
            ax3[1,1].scatter(paths, max_barriers_exp[i,:,1], s=2, alpha=0.1, facecolor='tab:orange')

            # formatting
            ax3[1,0].set_xlabel('Paths', fontsize=16)
            ax3[1,1].set_xlabel('Paths', fontsize=16)
            ax3[0,0].set_ylabel('$\Delta H_{M,i,max}^{\ddag}$ (kcal/mol)', fontsize=16)
            ax3[0,1].set_ylabel('$\Delta S_{M,i,max}^{\ddag}$ (kcal/mol/K)', fontsize=16)
            ax3[1,0].set_ylabel('$\Delta H_{M,i,max}^{\ddag}$ (kcal/mol)', fontsize=16)
            ax3[1,1].set_ylabel('$\Delta S_{M,i,max}^{\ddag}$ (kcal/mol/K)', fontsize=16)
            ax3[1,0].set_title('Exponentially distributed $\Delta H_{M,i,j}^{\ddag}$', fontsize=16)
            ax3[1,1].set_title('Exponentially distributed $\Delta S_{M,i,j}^{\ddag}$', fontsize=16)
            ax3[0,0].set_title('Normally distributed $\Delta H_{M,i,j}^{\ddag}$', fontsize=16)
            ax3[0,1].set_title('Normally distributed $\Delta S_{M,i,j}^{\ddag}$', fontsize=16)


    fig.savefig('figs/dH_dS_distributions.pdf')
    plt.show()


def show_maximums(dH_barrier, dS_barrier, dH_sigma, dS_sigma, T=300, multi=True):

    print(f'\nShowing maximum barriers across parallel paths...')

    title_dict = {'family' : 'Helvetica', 'size' : 8}

    n_paths = 2000
    fig1, ax1 = plt.subplots(1,1, figsize=(3.55,3.55/2))
    fig3, ax3 = plt.subplots(1,1, figsize=(3.55,2.5))

    # NORMAL DISTRIBUTION OF BARRIERS

    model = EyringModel(T=T)
    dist = 'normal'
    params = {'mu'  : np.array([dH_barrier, dS_barrier]),
              'cov' : np.array([[dH_sigma**2,0],
                                [0,dS_sigma**2]])}
    
    print(f'\tfor normal barriers:')

    # generate barriers and save the maximum barriers
    max_barriers = np.zeros(n_paths)
    for n in tqdm(range(n_paths)):
        model.add_Path(area=model.area/n_paths)
        model.paths[n].generate_membrane_barriers(dist=dist, multi=multi, dist_params=params)
        max_barriers[n] = model.paths[n].membrane_barriers.max()

    effective_barrier = model.calculate_effective_barrier()

    # plot the maximum barriers for arbitrarily numbered paths, plot effective barrier
    paths = np.arange(1, n_paths+1)
    ax1.scatter(paths, max_barriers, edgecolors='k', s=10, lw=0.5)
    ax1.axhline(effective_barrier, ls='dashed', c='k')
    xmin, xmax = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()
    ax1.text(xmax*0.75, effective_barrier-1, '$\Delta G_{eff}^{\ddag}$')

    # formatting
    ax1.set_ylabel('$\Delta G_{M,i,max}^{\ddag}$ (kcal/mol)')
    ax1.set_ylim(ymin-1, ymax)
    ax1.set_title('$\Delta H^{\ddag}_{M,i,j}$, $\Delta S^{\ddag}_{M,i,j}$ normally distributed', fontdict=title_dict)

    ax3.hist(max_barriers, edgecolor='k', bins=50, lw=0.5, density=True, facecolor='tab:blue')
    ax3.axvline(effective_barrier, ls='dashed', c='k')
    ax3.text(effective_barrier+0.4, 0.3, '$\Delta G_{eff}^{\ddag}$')
    # ax3.set_xlabel('Maximum $\Delta G_{M,i,j}^{\ddag}$ along path $i$ (kcal/mol)')
    ax3.set_ylabel('Probability')
    ax3.set_title('$\Delta H^{\ddag}_{M,i,j}$, $\Delta S^{\ddag}_{M,i,j}$ normally distributed', fontdict=title_dict)

    fig3.savefig('figs/maximum_barriers_normal.pdf')
    
    # EXPONENTIAL DISTRIBUTION OF BARRIERS

    fig2, ax2 = plt.subplots(1,1, figsize=(3.55,3.55/2))
    fig4, ax4 = plt.subplots(1,1, figsize=(3.55,2.5))

    model = EyringModel(T=T)
    dist = 'exponential'
    params = {'beta'  : np.array([dH_barrier, dS_barrier])}
    
    print(f'\tfor exponential barriers:')

    # generate barriers and save maximum barriers
    max_barriers = np.zeros(n_paths)
    for n in tqdm(range(n_paths)):
        model.add_Path(area=model.area/n_paths)
        model.paths[n].generate_membrane_barriers(dist=dist, multi=multi, dist_params=params)
        max_barriers[n] = model.paths[n].membrane_barriers.max()

    effective_barrier = model.calculate_effective_barrier()

    # plot maximum barriers and effective barrier
    paths = np.arange(1, n_paths+1)
    ax2.scatter(paths, max_barriers, edgecolors='k', c='tab:orange', s=10, lw=0.5)
    ax2.axhline(effective_barrier, ls='dashed', c='k')
    xmin, xmax = ax2.get_xlim()
    ymin, ymax = ax2.get_ylim()
    ax2.text(xmax*0.75, effective_barrier-8, '$\Delta G_{eff}^{\ddag}$')

    # formatting
    ax2.set_ylabel('$\Delta G_{M,i,max}^{\ddag}$ (kcal/mol)')
    ax2.set_ylim(ymin-10, ymax)
    ax2.set_title('$\Delta H^{\ddag}_{M,i,j}$, $\Delta S^{\ddag}_{M,i,j}$ exponentially distributed', fontdict=title_dict)
    ax2.set_xlabel('Paths')

    ax4.hist(max_barriers, edgecolor='k', bins=50, lw=0.5, density=True, facecolor='tab:orange')
    ax4.axvline(effective_barrier, ls='dashed', c='k')
    ax4.text(effective_barrier+2, 0.035, '$\Delta G_{eff}^{\ddag}$')
    ax4.set_xlabel('Maximum $\Delta G_{M,i,j}^{\ddag}$ along path $i$ (kcal/mol)')
    ax4.set_ylabel('Probability')
    ax4.set_title('$\Delta H^{\ddag}_{M,i,j}$, $\Delta S^{\ddag}_{M,i,j}$ exponentially distributed', fontdict=title_dict)

    fig4.savefig('figs/maximum_barriers_exponential.pdf')

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

    # sigs = np.array([0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 10])
    dH_sigs = np.array([0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10])
    dS_sigs = dH_sigs / T
    # dS_sigs = np.array([10e-5, 10e-4, 5e-4, 1e-4, 5e-3, 1e-3, 0.05, 0.01, 0.1, 0.5, 0.75, 0.9])
    n_sigs = len(dH_sigs)*len(dS_sigs)

    # save data per path for ROC curves
    perm_per_path = np.zeros(n_paths*n_sigs)
    perm_percent = np.zeros(n_paths*n_sigs)
    models_dH = np.zeros(n_paths*n_sigs)
    models_dS = np.zeros(n_paths*n_sigs)
    models = np.zeros(n_paths*n_sigs)

    # save data for overall model
    effective_barriers = np.zeros(n_sigs)
    permeabilities = np.zeros(n_sigs)
    max_barriers = np.zeros(n_sigs)
    max_enthalpies = np.zeros(n_sigs)
    max_entropies = np.zeros(n_sigs)
    dH_sigmas = np.zeros(n_sigs)
    dS_sigmas = np.zeros(n_sigs)
    i = 0
    for dH_sig in tqdm(dH_sigs):
        for dS_sig in dS_sigs:

            model = EyringModel(T=T)
            params = {'mu'  : np.array([dH_barrier, dS_barrier]),
                    'cov' : np.array([[dH_sig**2,0],
                                        [0,dS_sig**2]])}
    
            dH_max = -10e8
            dS_max = -10e8
            dG_max = -10e8
            for n in range(n_paths):
                model.add_Path(lam=10, area=model.area/n_paths)
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
    df_roc['paths'] = np.arange(1,n_paths+1).tolist()*n_sigs
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


def vary_everything(n_jumps_mu, jump_dist, jump_params, barrier_dist, barrier_params, n_paths=4, n_jumps_sig=3, T=300, plot=True):

    # generate a model with full variation in barriers, number of jumps, jump lengths
    model = EyringModel(T=T)
    for i in range(n_paths):

        # draw the number of jumps from a normal distribution
        n_jumps = int(np.random.default_rng().normal(loc=n_jumps_mu, scale=n_jumps_sig))

        # generate jump length distribution barrier distributions from normal distributions
        path = Path(T=T, n_jumps=n_jumps, lam=10)
        path.generate_jump_distribution(jump_dist, jump_params)
        path.generate_membrane_barriers(barrier_dist, barrier_params, multi=True)
        model.paths.append(path)
        model.deltas.append(path.jump_lengths.sum())
        
        # if any of the jump lengths are negative, set to 0.1 (should not be many with reasonable parameters for jump length distribution)
        for j in range(len(path.jump_lengths)):
            jump = path.jump_lengths[j]
            if jump < 0:
                path.jump_lengths[j] = 0.1

    model.n_paths = len(model.paths)
    dG_eff = model.calculate_effective_barrier() 

    # determine which path has the smallest maximum    
    min_max_barrier = 10e8
    for i in range(n_paths):
        mb = model.paths[i].membrane_barriers.max()
        if mb < min_max_barrier:
            min_max_path = model.paths[i]
            min_max_barrier = mb
            min_max_idx = i

    # determine the maximum permeability path
    P = model.calculate_permeability()
    max_perm_path = model.paths[model.permeabilities.argmax()]
    max_perm_path_barrier = max_perm_path.membrane_barriers.max()

    # plot the maximum permeability path and the smallest maximum barrier path
    if plot and min_max_idx != model.permeabilities.argmax(): # don't plot until smallest barrier path is not the most permeable path
        
        fig, ax = plt.subplots(1,1, figsize=(7.25,3.55))

        # first, for the path with the smallest maximum barrier
        jumps = min_max_path.jump_lengths.cumsum()
        barriers = min_max_path.membrane_barriers

        # fit the barrier profile with a cubic spline for smooth representation
        path_spline = CubicSpline(jumps, barriers, bc_type='natural')
        xs = np.linspace(0, jumps.max(), num=300)
        ys = path_spline(xs)
        ax.plot(xs, ys, c='r', label='smallest maximum barrier path')

        # plot the number of jumps in the path
        ax.text(jumps[-1]+5, barriers[-1], f'{len(jumps)} jumps', c='r')

        # and now for the max permeability path 
        jumps = max_perm_path.jump_lengths.cumsum()
        barriers = max_perm_path.membrane_barriers

        # fit the barrier profile with a cubic spline for smooth representation        
        path_spline = CubicSpline(jumps, barriers, bc_type='natural')
        xs = np.linspace(0, jumps.max(), num=300)
        ys = path_spline(xs)
        ys[0] = ys[1:10].mean()
        ax.plot(xs, ys, c='b', label='maximum permeability path')

        # plot the number of jumps in the path
        ax.text(jumps[-1]+5, barriers[-1], f'{len(jumps)} jumps', c='b')

        # plot the effective barrier shifted to account for entropic penalty
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        shifted_dG_eff = dG_eff + R*T*np.log(np.sum(model.areas) / model.area)
        ax.axhline(shifted_dG_eff, ls='dashed', c='black')
        ax.text(xmax*0.55, shifted_dG_eff+0.5, '$\Delta G_{eff}^{\ddag} + RT \ln(\sum_i^n A_i / A)$', ha='left', c='black')

        # formatting
        ax.set_xlabel('Transport Coordinate ($\mathrm{\AA}$)')
        ax.set_ylabel('$\Delta G^{\ddag}_{M,i,j}$')
        ax.set_ylim(ymin, ymax+3)
        ax.legend(frameon=False, ncol=3, loc='upper center')
        # ax.set_title('Free energy paths through membrane, normally distributed barriers, normally distributed jumps', fontsize=16)

        plt.savefig('figs/vary_everything_no_max_barriers.pdf')
        plt.show()

    # compare the difference in barrier heights to kT in kcal/mol
    return min_max_idx == model.permeabilities.argmax(), max_perm_path_barrier - min_max_barrier


def simulated_RO_v_NF(dH_RO, dS_RO, dH_NF, dS_NF, n_jumps=200, n_paths=2000, T=300, n_bootstraps=50):
    '''Similar to the experimental comparison between RO and NF barriers, calculate effective barriers for high mean, high variance (RO) model and low mean, high variance (NF) model'''

    # Numerical representation of RO membrane

    model_RO = EyringModel(T=T)
    dist = 'normal'
    params = {'mu'  : np.array([dH_RO, dS_RO]),
              'cov' : np.array([[5**2,0],
                                [0,5/T**2]])}
    
    m0 = params['mu'][0]
    m1 = params['mu'][1]
    s0 = np.sqrt(params['cov'][0,0])
    s1 = np.sqrt(params['cov'][1,1])
    
    # do a bootstrap to get error on effective barrier
    print('\nCalculating effective barrier for RO membrane...')
    effective_barriers_RO = np.zeros(n_bootstraps)
    for b in tqdm(range(n_bootstraps)):

        # membrane barrier distribution
        for n in range(n_paths):
            model_RO.add_Path(n_jumps=n_jumps, area=model_RO.area/n_paths)
            # model_RO.paths[n].generate_membrane_barriers(dist=dist, multi=True, dist_params=params)
            d0 = truncnorm((0-m0)/s0, (10**6-m0)/s0, loc=m0, scale=s0)
            dH = d0.rvs(n_jumps)
            d1 = truncnorm((-10**6-m1)/s1, (0-m1)/s1, loc=m1, scale=s1)
            dS = d1.rvs(n_jumps)

            model_RO.paths[n].enthalpic_barriers = dH
            model_RO.paths[n].entropic_barriers = dS
            model_RO.paths[n].membrane_barriers = dH - T*dS
            
        effective_barriers_RO[b] = model_RO.calculate_effective_barrier()

    print(f'RO effective barrier: {effective_barriers_RO.mean():.2f} +/- {effective_barriers_RO.std():.2f} kcal/mol')

    # Numerical representation of NF membrane

    model_NF = EyringModel(T=T)
    dist = 'normal'
    params = {'mu'  : np.array([dH_NF, dS_NF]),
              'cov' : np.array([[5**2,0],
                                [0,5/T**2]])}

    m0 = params['mu'][0]
    m1 = params['mu'][1]
    s0 = np.sqrt(params['cov'][0,0])
    s1 = np.sqrt(params['cov'][1,1])
        
    # do a bootstrap to get error on effective barrier
    print('\nCalculating effective barrier for NF membrane...')
    effective_barriers_NF = np.zeros(n_bootstraps)
    for b in tqdm(range(n_bootstraps)):

        # membrane barrier distribution
        for n in range(n_paths):
            model_NF.add_Path(n_jumps=n_jumps, area=model_NF.area/n_paths)
            # model_NF.paths[n].generate_membrane_barriers(dist=dist, multi=True, dist_params=params)
            d0 = truncnorm((0-m0)/s0, (10**6-m0)/s0, loc=m0, scale=s0)
            dH = d0.rvs(n_jumps)
            d1 = truncnorm((-10**6-m1)/s1, (0-m1)/s1, loc=m1, scale=s1)
            dS = d1.rvs(n_jumps)

            model_NF.paths[n].enthalpic_barriers = dH
            model_NF.paths[n].entropic_barriers = dS
            model_NF.paths[n].membrane_barriers = dH - T*dS
            
        effective_barriers_NF[b] = model_NF.calculate_effective_barrier()

    print(f'NF effective barrier: {effective_barriers_NF.mean():.2f} +/- {effective_barriers_NF.std():.2f} kcal/mol')

    return model_RO, model_NF


if __name__ == '__main__':

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'Helvetica'
    plt.rcParams['font.size'] = 8
    plt.rc('text.latex', preamble=r'\usepackage[cm]{sfmath}')

    # Inputs
    T = 300
    multi = True
    dH_barrier = 3.5
    dS_barrier = -9/T
    dH_sigma = dH_barrier/3
    dS_sigma = -dS_barrier/3
    n_paths = 2000 # infinite limit, approximately corresponds to unit area = 0.1 um^2
    
    avg_jumps = 40
    jump_dist = 'norm'
    jump_params = {'mu' : 10,
                   'sigma' : 2.5}
    barrier_dist = 'norm'
    barrier_params = {'mu' : np.array([dH_barrier, dS_barrier]),
                      'cov': np.array([[dH_sigma**2, 0],
                                       [0, dS_sigma**2]])}

    dG_barrier = dH_barrier - T*dS_barrier

    # Choose what analyses to run

    # Figure 2a
    # compare_effective_barriers(dH_barrier, dS_barrier, dH_sigma, dS_sigma, dG_barrier, T=T, multi=multi)
    
    # Figure 2b
    # plot_paths(1, dH_barrier, dS_barrier, dH_sigma, dS_sigma, T=T, multi=multi)

    # Figure 3
    # parallel_pores(dH_barrier, dS_barrier, dH_sigma, dS_sigma, dG_barrier, T=T, multi=multi, n_jumps=200, n_paths=2000, output='tmp')

    # Figure 4a,b
    show_maximums(dH_barrier, dS_barrier, dH_sigma, dS_sigma, T=T, multi=multi)
    exit()
    
    # Data for Figure 4c
    # barrier_variance(dH_barrier, dS_barrier, n_paths=n_paths, T=T)

    # Figure 5
    # is_equal = True
    # while is_equal:
    #     is_equal, diff = vary_everything(avg_jumps, jump_dist, jump_params, barrier_dist, barrier_params, n_paths=n_paths)

    ## Supplement to Figure 5: Calculate percentage of smallest max path == most permeable path
    # barrier_dist = 'exp'
    # barrier_params = {'beta' : np.array([dH_barrier, dS_barrier])}

    # n_iter = 3000
    # smallest_max_is_most_perm = 0
    # barrier_differences = np.zeros(n_iter)
    # for i in tqdm(range(n_iter)):
    #     is_equal, diff = vary_everything(avg_jumps, jump_dist, jump_params, barrier_dist, barrier_params, n_paths=2000, plot=False)
    #     if is_equal:
    #         smallest_max_is_most_perm += 1
    #     else:
    #         barrier_differences[i] = diff

    # barrier_differences = barrier_differences[barrier_differences != 0]
    # kT_cutoff = barrier_differences <= 1.987204259*10**-3*T

    # print(f'The path with the smallest maximum barrier is the most permeable path {smallest_max_is_most_perm/n_iter*100:.2f}% of {n_iter} iterations')
    # print(f'Of the {n_iter-smallest_max_is_most_perm} iterations where the smallest maximum barrier path is not the most permeable path, the maximum barrier of the most permeable path is within kT of the smallest maximum barrier {kT_cutoff.sum() / kT_cutoff.shape[0]*100:.2f}% of the time')

    # Figure 6
    # estimate_dH_dS(dH_barrier, dS_barrier, dH_sigma, dS_sigma, n_paths=22000, area=1e8, plot=True)    

    # Unused
    # fixed_jump_length(dH_barrier, dS_barrier, n_paths=n_paths, T=T, multi=multi)
    # RO, NF = simulated_RO_v_NF(dH_RO=4.6, dS_RO=-17.8/T, dH_NF=3.4, dS_NF=-17.9/T, n_paths=n_paths, T=T, n_bootstraps=1)
    # compare_jump_lengths(dH_barrier, dS_barrier, n_paths, delta=400, T=T, multi=multi)
    equal, normal, exponential = parallel_pores(dH_barrier, dS_barrier, dH_sigma, dS_sigma, dG_barrier, n_jumps=5, n_paths=2000, T=T, multi=multi, output='figs/parallel_pores_5jumps.png')

    plt.figure(figsize=(3.25, 2.6625))
    plt.xlabel('paths')
    plt.ylabel('flux fraction')

    print('\n-------------------------- NORMAL --------------------------')
    print(f'Overall barrier: {normal.calculate_effective_barrier():.4f}')
    print(f'Overall permeability: {normal.calculate_permeability():.4f}')

    flux_frac = np.zeros(normal.n_paths)
    for n,p in enumerate(normal.paths):
        flux_frac[n] = p.calculate_permeability() / normal.permeabilities.sum()

        print(f'Path {n}:')
        print(f'\tEffective barrier: {p.calculate_effective_barrier():.4f}')
        print(f'\tFlux fraction: {flux_frac[n]:.4e}')

    plt.scatter(np.arange(normal.n_paths), flux_frac, label='normal')

    print('\n-------------------------- EXPONENTIAL --------------------------')
    print(f'Overall barrier: {exponential.calculate_effective_barrier():.4f}')
    print(f'Overall permeability: {exponential.calculate_permeability():.4f}')

    flux_frac = np.zeros(exponential.n_paths)
    for n,p in enumerate(exponential.paths):
        flux_frac[n] = p.calculate_permeability() / exponential.permeabilities.sum()

        print(f'Path {n}:')
        print(f'\tEffective barrier: {p.calculate_effective_barrier():.4f}')
        print(f'\tFlux fraction: {flux_frac[n]:.4e}')

    plt.scatter(np.arange(exponential.n_paths), flux_frac, label='exponential')

    plt.legend()
    plt.show()