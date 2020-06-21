"""
The module implements path simulation for few models with rough volatility including rough Heston model
and rough SABR model. The main reference paper is "Functional Central Limit Theorems for Rough Volatility"
by Horvath et al.

Initial version of the code implemented by mgrillo, then refactored and standardized by atukallo
"""
import numpy as np

import functools
import py_vollib.black_scholes.implied_volatility as vollib_iv

from scipy import signal
from warnings import warn

from src.utils import *

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Heston ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Usual Heston model has the following form (interest rate and dividend rate are assumed to be zero):
d X_t = - V_t / 2 * dt + sqrt(V_t) * d W_t
V_t = Y_t
d Y_t = (theta - kappa * Y_t) * dt + sigma * sqrt(Y_t) * d B_t
d[W_t, B_t] = rho * dt
The model parameters are {theta, kappa, sigma, rho}, observables are {X_0, Y_0}.
Under such log price movement stock price follows
d S_t = sqrt(Y_t) * S_t * d W_t
For reference see https://nbviewer.jupyter.org/url/people.math.ethz.ch/~jteichma/lecture_ml_web/heston_calibration.ipynb

Rough Heston model is (interest rate and dividend rate are assumed to be zero):
d X_t = - V_t / 2 * dt + sqrt(V_t) * d W_t
V_t = (G^alpha Y)_t
d Y_t = (theta - kappa * Y_t) * dt + sigma * sqrt(Y_t) * d B_t
d[W_t, B_t] = rho * dt
where G^alpha is the Generalized Fractional Operator with alpha in (-1/2, 1/2).
G^alpha can be understood as an operator, which transforms the Hölder-1/2 function to Hölder-(1/2+alpha) function
and thus changes its 'roughness'.
To model parameters one more parameter {alpha} is added. Moreover, another observable V_0 is added, which
generally can be different from Y_0.
For reference see "Functional Central Limit Theorems for Rough Volatility" by Horvath et al and "Asymptotic 
Behaviour of the Fractional Heston Model" by Shi et al.
"""
# todo(mgrillo,atukallo): are both V_0 and Y_0 observables?


def simulate_rough_heston(n, m, terminal_time=1, log_spot_price=1, inst_vola=0.1, inst_vola_of_vola=0.1,
                          mean_rev_speed=1, eq_var=0.1, vola_of_vola=0.01, correlation=0.1, alpha=0.):
    """
    n is number of time steps
    m is number of simulations
    """
    T = terminal_time
    X_0 = log_spot_price
    V_0 = inst_vola
    Y_0 = inst_vola_of_vola
    kappa = mean_rev_speed
    theta = eq_var
    sigma = vola_of_vola
    rho = correlation

    warn('function is not tested yet, todo(mgrillo, atukallo)')
    # todo(mgrillo,atukallo): what should be assert on parameters to guarantee V > 0 ? Should further debug, not
    #   production ready yet.

    Y, B = __simulate_heston_volatility(n, m, T, Y_0, kappa, theta, sigma)
    V = __simulate_rough_volatility(n, m, T, V_0, alpha, Y)
    W = __get_correlated_bm(rho, B)
    X = __simulate_heston_log_price(n, m, T, X_0, V, W)

    return X


def __simulate_heston_volatility(n, m, T, Y_0, kappa, theta, sigma):
    """
    Simulates volatility process Y_t and returns it together with its brownian motion B_t
    """
    # todo(atukallo): use V_0 in code
    sqrtn = np.sqrt(n)
    vola = np.zeros((n, m))
    B = np.random.normal(0.0, 1.0, size=(n, m))
    vola[0] = Y_0
    for i in range(1, n):
        # todo(atukallo): why T / sqrtn ?
        vola[i] = vola[i - 1] \
                  + (theta - kappa * vola[i - 1]) * (T / n) \
                  + (sigma * np.sqrt(vola[i - 1])) * (T / sqrtn) * B[i - 1]
    return vola, B


def __simulate_rough_volatility(n, m, T, V_0, alpha, Y):
    assert -0.5 <= alpha <= 0.5, "alpha has disallowed value"

    V = np.zeros((n, m))
    g_vector = np.array([np.power(1. * T * i / n, alpha) for i in range(1, n)])  # dropping 0th element
    vola_incrs = Y[1:] - Y[:-1]

    V[0] = V_0
    # todo(mgrillo,atukallo): if becomes a bottle neck, add vectorization
    for i in range(m):
        # documentation recommends using fft only for big arrays
        if n > 500:
            # V[1:, i] = V_0 + signal.fftconvolve(g_vector, vola_incrs[:, i], 'same')
            V[1:, i] = V_0 + signal.fftconvolve(g_vector, vola_incrs[:, i], 'full')[0:n - 1]
        else:
            V[1:, i] = V_0 + signal.convolve(g_vector, vola_incrs[:, i], 'full')[0:n - 1]
    return V


def __simulate_heston_log_price(n, m, T, X_0, V, W):
    assert not np.any(np.isnan(V) | np.isnan(W))
    assert np.nanmin(V) > 0

    X = np.zeros((n, m))
    X[0] = X_0
    # todo(mgrillo,atukallo): code version is very different from the initial one, which is correct?
    for i in range(1, n):
        X[i] = X[i - 1] \
               - 0.5 * V[i - 1] * (T / n) \
               + np.sqrt(V[i - 1]) * np.sqrt(T / n) * W[i - 1]
    return X


def __get_correlated_bm(rho, B):
    W = rho * B + np.sqrt(1 - np.power(rho, 2)) * np.random.normal(0.0, 1.0, size=B.shape)
    return W


"""
Rough SABR model has the following form for log-price:
d X_t = - 0.5 * (V_t)^2 * L(t, X_t)^2 * dt + V_t * L(t, X_t) * d W_t
V_t = (G^alpha Y)_t
d Y_t = sigma * Y_t * d B_t
d[W_t, B_t] = rho * dt
Thus SABR model combines rough stochastic and local volatility
"""


def simulate_rough_SABR(n, m, terminal_time=1, log_spot_price=1, inst_vola=0.1, inst_vola_of_vola=0.1,
                        vola_of_vola=0.01, correlation=0.1, local_vola=lambda t, log_price: 1, alpha=0.):
    """
    n is number of time steps
    m is number of simulations
    """
    T = terminal_time
    X_0 = log_spot_price
    V_0 = inst_vola_of_vola
    Y_0 = inst_vola
    sigma = vola_of_vola
    rho = correlation
    L = local_vola

    Y, B = __simulate_SABR_volatility(n, m, T, Y_0, sigma)
    V = __simulate_rough_volatility(n, m, T, V_0, alpha, Y)
    W = __get_correlated_bm(rho, B)
    X = __simulate_SABR_log_price(n, m, T, X_0, L, V, W)

    if np.any(np.isnan(X)):
        warn(f'when simulating SABR got {np.count_nonzero(np.isnan(X))} nans!')

    return X


def get_option_prices_rough_SABR(n, m, maturity_times, moneynesses, log_spot_price=0, inst_vola=0.1,
                                 inst_vola_of_vola=0.1, vola_of_vola=0.01, correlation=0.1,
                                 local_vola=lambda t, log_price: 1, alpha=0., is_call_option=True):
    """
    n is number of time steps
    m is number of simulations
    """

    spot_price = np.exp(log_spot_price)
    strike_prices = [moneyness * spot_price for moneyness in moneynesses]

    terminal_time = np.max(maturity_times)
    log_stock = simulate_rough_SABR(n, m, terminal_time, log_spot_price, inst_vola, inst_vola_of_vola, vola_of_vola,
                                    correlation, local_vola, alpha)
    stock = np.exp(log_stock)

    # todo(mgrillo): maybe we don't want to use strike price, but instead moneyness

    option_price_f = call_price if is_call_option else put_price
    option_prices = np.zeros((len(maturity_times), len(strike_prices)))

    for i in range(len(maturity_times)):
        time_index = int(maturity_times[i] * n / np.max(maturity_times)) - 1
        for j in range(len(strike_prices)):
            option_prices[i, j] = np.nanmean(option_price_f(stock[time_index], strike_prices[j]))

    return option_prices


def get_implied_vola_rough_SABR(n, m, maturity_times, moneynesses, log_spot_price=0, inst_vola=0.1,
                                 inst_vola_of_vola=0.1, vola_of_vola=0.01, correlation=0.1,
                                 local_vola=lambda t, log_price: 1, alpha=0., is_call_option=True):

    option_prices = get_option_prices_rough_SABR(n, m, maturity_times, moneynesses, log_spot_price, inst_vola,
                                 inst_vola_of_vola, vola_of_vola, correlation, local_vola, alpha, is_call_option)

    # todo(atukallo): somehow often fails with "vola is below the intrinsic value"

    spot_price = np.exp(log_spot_price)
    strike_prices = [moneyness * spot_price for moneyness in moneynesses]
    implied_volas = np.zeros(option_prices.shape)
    flag = 'c' if is_call_option else 'p'
    for i, maturity in enumerate(maturity_times):
        for j, strike_price in enumerate(strike_prices):
            print(option_prices[i][j], spot_price, strike_price, maturity, flag)
            implied_volas[i][j] = vollib_iv.implied_volatility(price=option_prices[i][j], S=spot_price, K=strike_price,
                                                               t=maturity, r=0, flag=flag)
    return implied_volas


def __simulate_SABR_volatility(n, m, T, Y_0, sigma):
    """
    Simulates volatility process Y_t and returns it together with its brownian motion B_t
    """
    sqrt_step_size = np.sqrt(1. * T / n)
    vola = np.zeros((n, m))
    B = np.random.normal(0.0, 1.0, size=(n, m))
    vola[0] = Y_0
    for i in range(1, n):
        # todo(atukallo,mgrillo): why T / sqrtn ? (see paper page 12, 2nd step of Algo 3.3, question forwarded to Wahid)
        vola[i] = vola[i - 1] \
                  + (sigma * np.sqrt(vola[i - 1])) * sqrt_step_size * B[i - 1]
    return vola, B


def __simulate_SABR_log_price(n, m, T, X_0, L, V, W):
    assert not np.any(np.isnan(V) | np.isnan(W))

    X = np.zeros((n, m))
    X[0] = X_0
    for i in range(1, n):
        binded_L = np.vectorize(functools.partial(L, T * (i - 1) / n))
        X[i] = X[i - 1] \
               - 0.5 * np.power(V[i - 1], 2) * np.power(binded_L(X[i - 1]), 2) * (T / n) \
               + V[i - 1] * np.sqrt(T / n) * W[i - 1]
    return X
