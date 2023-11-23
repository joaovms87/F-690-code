import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
import scipy.linalg as sl
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




g2 = 1      # gamma2

DELTA = 2  # total variation of gamma1
gamma1_0 = 0.1
gamma1_f = 2.1


# eigen-energies as functions of gamma1 = g
def E1(g):
    return -np.sqrt((g+g2)**2 + 1)/2


def E2(g):
    return -np.sqrt((g-g2)**2 + 1)/2


def E3(g):
    return np.sqrt((g-g2)**2 + 1)/2


def E4(g):
    return np.sqrt((g+g2)**2 + 1)/2


def partition(g):
    e1, e2, e3, e4 = E1(g), E2(g), E3(g), E4(g)
    return np.exp(-beta*e1) + np.exp(-beta*e2) + np.exp(-beta*e3) + np.exp(-beta*e4)


def W_ad():
    E0 = [E1(gamma1_0), E2(gamma1_0), E3(gamma1_0), E4(gamma1_0)]
    Ef = [E1(gamma1_f), E2(gamma1_f), E3(gamma1_f), E4(gamma1_f)]
    sum = 0
    for i in range(4):
        sum += np.exp(-beta*E0[i])*(Ef[i]-E0[i])
    return sum/partition(gamma1_0)


def DeltaF():
    Z0 = partition(gamma1_0)
    Zf = partition(gamma1_f)
    return -beta**(-1)*np.log(Zf/Z0)


betapoints = np.logspace(-4, 2, 120)
taupoints = np.logspace(-1, 3.5, 100)
l_taupoints = list(taupoints)
i_med = l_taupoints.index(1)

DELTA = 2
# take as reference the data for the smaller beta
D_ref = np.real(np.loadtxt(f'D beta=0.0001 DELTA={DELTA}', complex))
n = len(D_ref)

for j, beta in enumerate(betapoints):
    if beta > 0.1:
        break


titles = [r'$\beta$-dependence of $D[\rho||\rho_{eq}]$ for small $\tau$',
          r'$\beta$-dependence of $D[\rho||\rho_{eq}]$ for $\tau J/\hbar=1$',
          r'$\beta$-dependence of $D[\rho||\rho_{eq}]$ close to the adiabatic limit']
tau_list = ['$10^{-1}$', '$10^0$', '$10^{3.5}$']
files = ['fast', 'medium', 'adiabatic']
for k, index in enumerate([0, i_med, n-1]):
    D_list = []
    fig, ax = plt.subplots()
    label=r'$D[\rho||\rho_{eq}]$ for $\tau J/\hbar=$'+tau_list[k]
    for beta in betapoints:
        D = np.real(np.loadtxt(f'D beta={beta} DELTA={DELTA}', complex))[index]
        D_list.append(D)
    fit, cov = np.polyfit(np.log(betapoints[:j]), np.log(D_list[:j]), deg=1, cov=True)
    print(ufloat(fit[0], np.sqrt(cov[0,0])))
    D_fit = betapoints[:j]**fit[0] * np.exp(fit[1])

    ax.loglog(betapoints, D_list, 'k.', label=label)
    ax.loglog(betapoints[:j], D_fit, 'r-', label=r'Fit for $\beta J<0.1$')

    ax.legend(fancybox=False, edgecolor='black', loc='lower right')
    ax.set_xlabel(r'$\beta\cdot J$')
    ax.set_ylabel(r'$D[\rho||\rho_{eq}]$')
    ax.set_title(titles[k])
    ax.set_box_aspect(0.8)
    ax.grid(True)
    plt.savefig(f'D:\GRÁFICOS IC\spins2\RK beta_dependence\V3/D {files[k]}', bbox_inches='tight', dpi=400)

fig2, ax = plt.subplots()
colors = ['red', 'darkgoldenrod', 'green', 'blue', 'black']
labels = [r'$\beta J = 10^{-3}$', r'$\beta J = 10^{-2}$', r'$\beta J = 10^{-1}$', r'$\beta J = 10^{0}$', r'$\beta J = 10^{1}$']
for k, beta in enumerate([0.001, 0.01, 0.1, 1, 10]):
    data = np.real(np.loadtxt(f'D beta={beta} DELTA=2', complex))
    ax.loglog(taupoints, data, color=colors[k], marker='.', label=labels[k], lw=0)

ax.set_title(r'$D[\rho||\rho_{eq}]$ as a function of $\tau$ for several $\beta$')
ax.set_box_aspect(0.8)
ax.set_xlabel(r'$\tau J/\hbar$')
ax.set_ylabel(r'$D[\rho||\rho_{eq}]$')
ax.grid(True)
ax.legend(fancybox=False, edgecolor='black', loc='upper right', fontsize=10, ncols=3)
ax.set_xlim(0.1, 10**3.5)
ax.set_ylim(top=1e3)
plt.savefig('D:\GRÁFICOS IC\spins2\RK beta_dependence\V3/all_D', bbox_inches='tight', dpi=400)


D_list = []
fig3, ax = plt.subplots()
for k, beta in enumerate(betapoints):
    D_list.append(beta*(W_ad()-DeltaF()))
fit, cov = np.polyfit(np.log(betapoints[:j]), np.log(D_list[:j]), deg=1, cov=True)
print(ufloat(fit[0], np.sqrt(cov[0,0])))
D_fit = betapoints[:j]**fit[0] * np.exp(fit[1])

ax.loglog(betapoints, D_list, 'k.', label=r'$D[\rho||\rho_{eq}]$ in the adiabatic limit')
ax.loglog(betapoints[:j], D_fit, 'r-', label=r'Fit for $\beta J<0.1$')

ax.legend(fancybox=False, edgecolor='black', loc='lower right')
ax.set_xlabel(r'$\beta\cdot J$')
ax.set_ylabel(r'$D[\rho||\rho_{eq}]$')
ax.set_title(r'$D[\rho||\rho_{eq}]$ in the adiabatic limit')
ax.set_box_aspect(0.8)
ax.grid(True)
plt.savefig(f'D:\GRÁFICOS IC\spins2\RK beta_dependence\V3/D_adiabatic_exact', bbox_inches='tight', dpi=400)

