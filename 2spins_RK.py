import numpy as np
import scipy.linalg as sl


g2 = 1      # gamma2

DELTA = 2  # total variation of gamma1
gamma1_0 = 0.1


def gamma1(t):  # protocol for gamma1 variation
    if t==0:
        return gamma1_0
    else:
        return gamma1_0 + DELTA/tau*t


# eigen-energies as functions of gamma1 = g
def E1(g):
    return -np.sqrt((g+g2)**2 + 1)/2


def E2(g):
    return -np.sqrt((g-g2)**2 + 1)/2


def E3(g):
    return np.sqrt((g-g2)**2 + 1)/2


def E4(g):
    return np.sqrt((g+g2)**2 + 1)/2


def partition(t):
    g = gamma1(t)
    e1, e2, e3, e4 = E1(g), E2(g), E3(g), E4(g)
    return np.exp(-beta*e1) + np.exp(-beta*e2) + np.exp(-beta*e3) + np.exp(-beta*e4)


def rho_eq(t):
    Z, g = partition(t), gamma1(t)
    e1, e2, e3, e4 = E1(g), E2(g), E3(g), E4(g)

    v1 = np.array([g+g2-2*e1, 0, 0, 1])
    v1 /= np.linalg.norm(v1)
    v2 = np.array([0, -g+g2-2*e2, 1, 0])
    v2 /= np.linalg.norm(v2)
    v3 = np.array([0, -g+g2-2*e3, 1, 0])
    v3 /= np.linalg.norm(v3)
    v4 = np.array([g+g2-2*e4, 0, 0, 1])
    v4 /= np.linalg.norm(v4)

    p = np.exp(-beta*e1)*np.outer(v1, v1)
    p+= np.exp(-beta*e2)*np.outer(v2, v2)
    p+= np.exp(-beta*e3)*np.outer(v3, v3)
    p+= np.exp(-beta*e4)*np.outer(v4, v4)

    return p/Z


def rho_ad(t):
    Z, g, g0 = partition(0), gamma1(t), gamma1_0
    e1, e2, e3, e4 = E1(g), E2(g), E3(g), E4(g)
    e10, e20, e30, e40 = E1(gamma1_0), E2(gamma1_0), E3(gamma1_0), E4(gamma1_0)

    v1 = np.array([g+g2-2*e1, 0, 0, 1])
    v1 /= np.linalg.norm(v1)
    v2 = np.array([0, -g+g2-2*e2, 1, 0])
    v2 /= np.linalg.norm(v2)
    v3 = np.array([0, -g+g2-2*e3, 1, 0])
    v3 /= np.linalg.norm(v3)
    v4 = np.array([g+g2-2*e4, 0, 0, 1])
    v4 /= np.linalg.norm(v4)

    p = np.exp(-beta * e10) * np.outer(v1, v1)
    p+= np.exp(-beta * e20) * np.outer(v2, v2)
    p+= np.exp(-beta * e30) * np.outer(v3, v3)
    p+= np.exp(-beta * e40) * np.outer(v4, v4)

    return p / Z


def Ham(t):
    g = gamma1(t)
    H_t = np.zeros([4, 4], complex)
    H_t[3, 0], H_t[2, 1], H_t[1, 2], H_t[0, 3] = -1, -1, -1, -1
    H_t[0, 0], H_t[1, 1], H_t[2, 2], H_t[3, 3] = -(g+g2), g-g2, -g+g2, g+g2
    return H_t/2


def D(p1, p2):  # relative entropy between states p1 and p2
    ln1, ln2 = sl.logm(p1), sl.logm(p2)
    prod1, prod2 = np.matmul(p1, ln1), np.matmul(p1, ln2)
    return np.trace(prod1-prod2)


def D_eq(p, t):
    deltaF = -1/beta*np.log(partition(t)/partition(0))
    return beta*(W(p)-deltaF)


def W(p):  # calculates total work done in the process
    Ef = np.trace(np.matmul(p, Ham(tau)))
    return np.real(Ef - E0)


def W_ad(t):  # calculates the work done up to time t if the evolution is adiabatic
    Ef = np.trace(np.matmul(rho_ad(t), Ham(t)))
    return np.real(Ef - E0)


def f(r, t):  # array of functions at the right-hand-side of the simultaneous equations
    g = gamma1(t)
    p00, p01, p02, p03, p11, p12, p13, p22, p23, p33 = r
    f00 = -1j * (p03 - np.conj(p03))
    f01 = -1j * (-2*g*p01 + p02 - np.conj(p13))
    f02 = -1j * (-2*g2*p02 + p01 - np.conj(p23))
    f03 = -1j * (-2*(g+g2)*p03 + p00 - p33)
    f11 = -1j * (p12 - np.conj(p12))
    f12 = -1j * (2*(g-g2)*p12 + p11 - p22)
    f13 = -1j * (-2*g2*p13 + np.conj(p01) - p23)
    f22 = -1j * (np.conj(p12) - p12)
    f23 = -1j * (-2*g*p23 + np.conj(p02) - p13)
    f33 = -1j * (np.conj(p03) - p03)

    return np.array([f00, f01, f02, f03, f11, f12, f13, f22, f23, f33], complex)/2


# The next functions are for the adaptive Runge-Kutta method
def ratio(a, b):
    aux = (a-b)/30
    eps = np.linalg.norm(aux)
    return h*delta/eps


def new_h(h, p):
    h_new = h*p**0.25
    if h_new>2*h:
        return 2*h
    else:
        return h_new


# list of the temperature parameters that will be investigated
betapoints = np.logspace(-4, 2, 120)
# list of the process durations that will be investigated
taupoints = np.logspace(-1, 3.5, 100)
l = len(taupoints)
# lists of total work and final D - each entry corresponds to a process with tau in taupoints:
Wpoints = np.empty(l, complex)
WADpoints = np.empty(l, complex)
Dpoints = np.empty(l, complex)    # D[rho(tau)||rho_eq(tau)]
D2points = np.empty(l, complex)   # D[rho(tau)||rho_ad(tau)]
# compute these quantities for each tau

delta = 1e-12  # required accuracy per unit time

for beta in betapoints:
    print(beta)
    # array of rho elements at time 0; necessary to apply the adaptive Runge-Kutta method
    rho0 = rho_eq(0)
    r0 = np.array([rho0[0, 0], rho0[0, 1], rho0[0, 2], rho0[0, 3],
                   rho0[1, 1], rho0[1, 2], rho0[1, 3],
                   rho0[2, 2], rho0[2, 3],
                   rho0[3, 3]], complex)
    for j, tau in enumerate(taupoints):
        t = 0
        N = 100
        h = tau/N  # Initial step size

        # Solve system of differential equations by the adaptive Runge-Kutta method
        r = r0.copy()
        while t<tau:
            # two steps of size h
            k1 = h*f(r, t)
            k2 = h*f(r+0.5*k1, t+0.5*h)
            k3 = h*f(r+0.5*k2, t+0.5*h)
            k4 = h*f(r+k3, t+h)
            r1 = r + (k1+2*k2+2*k3+k4)/6
            k1 = h*f(r1, t+h)
            k2 = h*f(r1+0.5*k1, t+1.5*h)
            k3 = h*f(r1+0.5*k2, t+1.5*h)
            k4 = h*f(r1+k3, t+2*h)
            r1 += (k1+2*k2+2*k3+k4)/6
            # one step of size 2h
            k1 = 2*h*f(r, t)
            k2 = 2*h*f(r+0.5*k1, t+h)
            k3 = 2*h*f(r+0.5*k2, t+h)
            k4 = 2*h*f(r+k3, t+2*h)
            r2 = r + (k1+2*k2+2*k3+k4)/6

            p = ratio(r1, r2)
            if p>=1:  # precision greater than required
                r = r1  # keep the most precise result
                t = t+2*h  # next t
            # if p>=1 is not satisfied, repeat everything without going to next t
            h = new_h(h, p)  # changes h regardless of the value of p


        # build the density matrix at time tau for the process of duration tau
        rho = np.empty([4, 4], complex)
        rho[0, 0], rho[0, 1], rho[0, 2], rho[0, 3], rho[1, 1], rho[1, 2], rho[1, 3], rho[2, 2], rho[2, 3], rho[3, 3] = r
        rho[1, 0], rho[2, 0], rho[3, 0], rho[2, 1], rho[3, 1], rho[3, 2] = \
            np.conj(rho[0, 1]), np.conj(rho[0, 2]), np.conj(rho[0, 3]), np.conj(rho[1, 2]), \
            np.conj(rho[1, 3]), np.conj(rho[2, 3])

        # initial energy for computation of the work:
        E0 = np.trace(np.matmul(rho0, Ham(0)))
        # compute the work done in the process of duration tau and save it
        Wpoints[j] = W(rho)
        # compute the relative entropy and save it
        Dpoints[j] = D_eq(rho, tau)
        # compute D[rho(tau)||rho_ad(tau)] and save it
        D2points[j] = D(rho, rho_ad(tau))
        # compute W_ad(tau) and save it
        WADpoints[j] = W_ad(tau)


    # save everything
    np.savetxt(f'W beta={beta} DELTA={DELTA}', Wpoints)
    np.savetxt(f'W_ad beta={beta} DELTA={DELTA}', WADpoints)
    np.savetxt(f'D beta={beta} DELTA={DELTA}', Dpoints)
    np.savetxt(f'D2 beta={beta} DELTA={DELTA}', D2points)
