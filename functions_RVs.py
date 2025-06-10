import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import emcee
from scipy.stats import norm
from numba import njit

def plotparams(ax, labelsize=15):
    '''
    Basic plot params

    :param ax: axes to modify

    :type ax: matplotlib axes object

    :returns: modified matplotlib axes object
    '''
    ax.minorticks_on()
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(direction='in', which='both', labelsize=labelsize)
    ax.tick_params('both', length=8, width=1.8, which='major')
    ax.tick_params('both', length=4, width=1, which='minor')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    return ax



def log_prior(theta):
    P, K, e, gamma, omega, T0 , logs = theta[:7]
    offsets = theta[7:]
    s = np.exp(logs)
    
    if not (0.1 < e < 0.99):
        return -np.inf
    if not (0 <= gamma <= 40):
        return -np.inf
    if not (-180 <= omega <= 180):
        return -np.inf
    if not (2.4213e6 <= T0 <= 2.4218e6):
        return -np.inf
    if not (-10 < logs < 0):
        return -np.inf
        

    log_prior = np.sum(norm.logpdf(offsets, loc=0, scale=1.0))  # σ = 1.0
    return log_prior

    
    return 0.0

# Likelihood

def log_likelihood(theta, t, rv_obs, rv_err):
    P, K, e, gamma, omega, T0 , logs = theta[:7]
    offsets = theta[7:]
    s = np.exp(logs)

    
    lnlike = 0
    
    for i in range(len(t)):
        times_i = t[i]
        rvs_i = rv_obs[i]
        errs_i = rv_err[i]

        if i == 4:
            model_i = keplerian_model(times_i, P, K, e, gamma, omega, T0)
        else:
            model_i = keplerian_model(times_i, P, K, e, gamma, omega, T0) + offsets[i-1]
    
        lnlike_i = -0.5 * np.sum(((rvs_i - model_i) / (errs_i**2+s**2)**(1/2)) ** 2 + np.log(2 * np.pi * (errs_i**2+s**2)))
        lnlike += lnlike_i

    return lnlike


def log_probability(theta, t, rv_obs, rv_err):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, t, rv_obs, rv_err)




# Functions
def compute_M2(fM, M1, i_rad): # check this
    def mass_func_eq(M2):
        return (M2 * np.sin(i_rad))**3 - fM * (M1 + M2)**2
    sol = root_scalar(mass_func_eq, bracket=[0.01, 10], method='bisect')
    return sol.root if sol.converged else None

def sma_f(T, M1,M2):
    """
    Calculates the semi-major axis using Kepler's Third Law.

    Args:
        T: Orbital period.
        mu: Standard gravitational parameter (GM).

    Returns:
        The semi-major axis.
    """
    mu = 6.674e-11*(M1+M2)*1.989e30
    a1 = (mu * ((T*24*60*60)**2) / (4 * np.pi**2))**(1/3)
    a = a1/1.496e+11

    return a

def compute_TI(sma, i, omega, lan):
    A = sma * (np.cos(lan)*np.cos(omega) - np.cos(i)*np.sin(lan)*np.sin(omega))
    B = sma * (np.sin(lan)*np.cos(omega) + np.cos(i)*np.cos(lan)*np.sin(omega))
    F = sma * (-np.cos(lan)*np.sin(omega) - np.cos(i)*np.sin(lan)*np.cos(omega))
    G = sma * (-np.sin(lan)*np.sin(omega) + np.cos(i)*np.cos(lan)*np.cos(omega))
    return A, B, F, G

def proj_sep(A, B, F, G, E, e):
    x = -A * (np.cos(E) - e) - F * np.sqrt(1 - e**2) * np.sin(E) #verify this is typed in correctly
    y = -B * (np.cos(E) - e) - G * np.sqrt(1 - e**2) * np.sin(E)
    return x, y