import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
plt.rcParams['errorbar.capsize'] = 3
plt.rcParams['mathtext.fontset'] = 'cm'
# plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['grid.color'] = 'grey'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['font.size'] = 12
plt.rcParams['legend.fontsize'] = 'large'
plt.rcParams['figure.titlesize'] = 'medium'

plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


betapoints = np.logspace(-4, 2, 120)
taupoints = np.logspace(-1, 3.5, 100)
l_taupoints = list(taupoints)
i_med = l_taupoints.index(1)

DELTA = 2
# take as reference the data for the smaller beta
W_ref = np.real(np.loadtxt(f'W beta=0.0001 DELTA={DELTA}', complex))
n = len(W_ref)

for j, beta in enumerate(betapoints):
    if beta > 0.1:
        break

for i, tau in enumerate(taupoints):
    if tau > 100:
        break



titles = [r'$\beta$-dependence of $W_{ex}$ for small $\tau$',
          r'$\beta$-dependence of $W_{ex}$ for $\tau J/\hbar=1$',
          r'$\beta$-dependence of $W_{ex}$ close to the adiabatic limit']
tau_list = ['$10^{-1}$', '$10^0$', '$10^{3.5}$']
files = ['fast', 'medium', 'adiabatic']
for k, index in enumerate([0, i_med, n-1]):
    Wex_list = []
    fig, ax = plt.subplots()
    label=r'$W_{ex}$ for $\tau J/\hbar=$'+tau_list[k]
    for beta in betapoints:
        W = np.real(np.loadtxt(f'W beta={beta} DELTA={DELTA}', complex))
        Wad = np.real(np.loadtxt(f'W_ad beta={beta} DELTA={DELTA}', complex))
        Wex = W - Wad
        if k==2:
            fit1, cov1 = np.polyfit(np.log(taupoints[i:]), np.log(Wex[i:]), deg=1, cov=True)
            # print(ufloat(fit1[0], np.sqrt(cov1[0,0])))
            c = np.exp(fit1[1])
            Wex_list.append(c)
        else:
            Wex_list.append(Wex[index])
    fit, cov = np.polyfit(np.log(betapoints[:j]), np.log(Wex_list[:j]), deg=1, cov=True)
    print(ufloat(fit[0], np.sqrt(cov[0,0])))
    Wex_fit = betapoints[:j]**fit[0] * np.exp(fit[1])

    if k==2:
        ax.loglog(betapoints, Wex_list, 'k.', label='$c$ values')
        ax.set_ylabel('$c$')
    else:
        ax.loglog(betapoints, Wex_list, 'k.', label=label)
        ax.set_ylabel(r'$J^{-1}W_{ex}$')

    ax.loglog(betapoints[:j], Wex_fit, 'r-', label=r'Fit for $\beta J<0.1$')

    ax.legend(fancybox=False, edgecolor='black', loc='lower right')
    ax.set_xlabel(r'$\beta\cdot J$')

    ax.set_title(titles[k])
    ax.set_box_aspect(0.8)
    ax.grid(True)
    plt.savefig(f'D:\GRÁFICOS IC\spins2\RK beta_dependence\V3/W {files[k]}', bbox_inches='tight', dpi=400)


fig2, ax = plt.subplots()
colors = ['red', 'darkgoldenrod', 'green', 'blue', 'black']
labels = [r'$\beta J = 10^{-3}$', r'$\beta J = 10^{-2}$', r'$\beta J = 10^{-1}$', r'$\beta J = 10^{0}$', r'$\beta J = 10^{1}$']
for k, beta in enumerate([0.001, 0.01, 0.1, 1, 10]):
    W = np.real(np.loadtxt(f'W beta={beta} DELTA=2', complex))
    Wad = np.real(np.loadtxt(f'W_ad beta={beta} DELTA=2', complex))
    Wex = W - Wad
    ax.loglog(taupoints, Wex, color=colors[k], marker='.', label=labels[k], lw=0)

ax.set_title(r'$W_{ex}$ as a function of $\tau$ for several $\beta$')
ax.set_box_aspect(0.8)
ax.set_xlabel(r'$\tau J/\hbar$')
ax.set_ylabel(r'$J^{-1}W_{ex}$')
ax.set_xlim(0.1, 10**3.5)
ax.grid(True)
ax.legend(fancybox=False, edgecolor='black', loc='lower left')
plt.savefig('D:\GRÁFICOS IC\spins2\RK beta_dependence\V3/all_Wex', bbox_inches='tight', dpi=400)
