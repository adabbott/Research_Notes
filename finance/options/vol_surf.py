import yfinance as yf
import numpy as np
import jax.numpy as jnp
#from jax.config import config; config.enable
import pandas as pd
import matplotlib.pyplot as plt
import urllib
import requests
import re
from datetime import datetime, timedelta

from black_scholes import implied_vol
from risk_free_rate import get_rfr

def find_iv(row, mode=0):
    """
    Wrapper for pandas apply to find IV given market option prices
    """
    try:
        result = implied_vol(row['price'], row['underlying'], row['strike'], row['tenor'], row['rate'], mode=mode) 
    except:
        result = 0.0
    return result

def vol_surf(ticker, mode=0):
    """
    Plots volatility surface using Black-Scholes
    ticker : string
        Ticker symbol; "^SPX", "AAPL", etc
    mode : int
        mode = 0 for calls, 1 for puts
    """
    # Risk free rate as function of time in years.
    print("Getting risk-free rates...")
    rfr_func = get_rfr()
    # Load ticker data and expiries
    print("Pulling underlying info...")
    spx = yf.Ticker(ticker)
    expiries = spx.options
    info = spx.info
    underlying_price = (info['bid'] + info['ask']) * 0.5
    print("Pulling options chain data...")
    option_chains = []
    for expiry in expiries:
        # time til expiration
        tte = (datetime.strptime(expiry, "%Y-%m-%d") - datetime.now()) / timedelta(days=365)
        # Short term vol surface only, <1yr
        if tte < 1.0 and tte > 0.0: 
            call_chain, put_chain = spx.option_chain(expiry)
            if mode == 0:
                chain = call_chain
            else:
                chain = put_chain
            # Remove strikes which differ from underlying price outside range -50% -> 50% 
            mask = (underlying_price * 0.5 < chain['strike']) & (underlying_price * 1.5 > chain['strike'])
            chain = chain.loc[mask]
            # Add time til expiry in years
            chain = chain.assign(tenor=np.repeat(tte, chain.shape[0]))
            # Add risk-free-rate for time till expiry
            rate = rfr_func(tte)
            chain = chain.assign(rate=np.repeat(rate, chain.shape[0]))
            option_chains.append(chain)

    data = pd.concat(option_chains) 
    # Add mid price
    price = data.apply(lambda row: 0.5 * (row['bid'] + row['ask']), axis=1)
    data = data.assign(price=price)
    # Add underlying
    data = data.assign(underlying=np.repeat(underlying_price, data.shape[0]))
    # Compute IV's
    print("Computing implied volatilities...")
    iv = data.apply(lambda row: find_iv(row, mode), axis=1)
    dataset = np.vstack((data['tenor'].values, data['strike'].values, iv.values)).T
    # Remove 0 IV rows (opt failed)
    dataset = dataset[np.all(dataset > 0.0, axis=1)]
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title("{} volatility surface".format(ticker))
    ax.set_xlabel('Tenor (years)')
    ax.set_ylabel('Strike')
    ax.set_zlabel('IV')
    ax.scatter3D(dataset[:,0], dataset[:,1], dataset[:,2])
    plt.show()

vol_surf("^SPX", mode=0)
#vol_surf("AAPL", mode=0)



