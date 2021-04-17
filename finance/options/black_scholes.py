import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax.scipy.optimize import minimize
from scipy import optimize 
from functools import partial
from datetime import datetime, timedelta 
# Note: enter greeks in vim with <ctrl> + k, letter, *

def yte(datestring, syntax="%m/%d/%Y"):
    """
    Years till expiry given expiration

    datestring : Example: "1/21/2022" 
    syntax : alternative form for providing datestring, describes how to parse month, day, year
    """
    expiry = datetime.strptime(datestring, syntax)
    yte = (expiry - datetime.now()) / timedelta(days=365)
    return yte

# Assumptions of BS: 
# return is random walk (brownian motion) with constant drift and volatility,
# assumes spot ~ lognormal dist, no tax/txn costs, rfr constant across all maturities, short selling is permitted, no arbitrage   
# also assumes continuously tradable ( no perturbation due to holiday/weekend theta decay)

@jax.jit
def black_scholes(σ, S, K, T, r = 0.0, mode=0):
    """
    σ : implied volatility
    S : underlying price 
    K : strike price 
    T : time till expiry in years
    r : risk-free rate
    mode : 0 for call  1 for put 
    """
    d1 = (jnp.log(S / K) + (r + σ**2 / 2) * T) / (σ * jnp.sqrt(T))
    d2 = d1 - σ * jnp.sqrt(T) 
    call = norm.cdf(d1) * S - norm.cdf(d2) * K * jnp.exp(-r * T)

    # For jit-compiling both put and call logic branches: 
    # Semantically the same as `if mode == 0: get call premium elif mode == 1 get put premium` 
    premium = jax.lax.cond(mode == 0, lambda _: call, lambda _: K * jnp.exp(-r * T) - S + call, mode)
    return premium

def implied_vol(P, S, K, T, r = 0.0, mode=0):
    """
    Numerically back-out implied volatility given market price of options contract
    Jit-compilation of black_scholes helps here 
    """
    def objective(σ):
        return black_scholes(σ, S, K, T, r, mode) - P
    return round(optimize.brentq(objective, 0., 10.), 5)

# Greeks with autodiff
bs_delta = jax.jacfwd(black_scholes, 1)
bs_gamma = jax.jacfwd(bs_delta, 1)
bs_theta = jax.jacfwd(black_scholes, 3)
bs_vega = jax.jacfwd(black_scholes, 0)
bs_vanna = jax.jacfwd(bs_delta, 0)
bs_charm = jax.jacfwd(bs_delta, 3)

# SPX options
# Data from: https://www.barchart.com/stocks/quotes/$SPX/volatility-greeks?expiration=2022-05-21-m
# Ignoring weekend drift...
# Take quoted IV and predict premium, greeks
args = (0.1218, 4185.47, 4200, yte("5/21/2021"))
p = black_scholes(*args) 
print(p)
δ = bs_delta(*args)
print(δ)
γ = bs_gamma(*args)
print(γ)
θ = bs_theta(*args) / 365
print(θ)
v = bs_vega(*args) / 100
print(v)
V = bs_vanna(*args)
print(V)
c = bs_charm(*args)
print(c)

# Take last trade premium, 54.30, backout IV, then compute greeks
iv = implied_vol(54.30, 4185.47, 4200, yte("5/21/2021"))
args = (iv, 4185.47, 4200, yte("5/21/2021"))
p = black_scholes(*args) 
print(p)
δ = bs_delta(*args)
print(δ)
γ = bs_gamma(*args)
print(γ)
θ = bs_theta(*args) / 365
print(θ)
v = bs_vega(*args) / 100
print(v)
V = bs_vanna(*args)
print(V)
c = bs_charm(*args)
print(c)

# Ideas, notes, thoughts:
# Plot black-scholes "error surface", i.e. difference in last traded price and predicted BS price 
# Front month vol skew
# Try implementing IV model, or statistical forecasts of rvol
# maybe this one https://www.tandfonline.com/doi/full/10.1080/14697688.2019.1675898
