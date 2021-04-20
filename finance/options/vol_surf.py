#from pandas_datareader import data as pdr 
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib
import requests
import re
from datetime import datetime, timedelta
#from GPy.models import GPRegression
#from GPy.kern import RBF
#import peslearn

from black_scholes import implied_vol
from risk_free_rate import get_rfr

def find_iv(row, rate, mode=0):
    """
    Wrapper for pandas apply to find IV given market option prices
    """
    try:
        result = implied_vol(row['price'], row['underlying'], row['strike'], row['tenor'], rate, mode=mode) 
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
    print("Pulling options chain data and computing IV...")

    # Will hold tenor, strike, IV pairs 
    dataset = []
    for expiry in expiries:
        # time til expiration
        tte = (datetime.strptime(expiry, "%Y-%m-%d") - datetime.now()) / timedelta(days=365)
        # Short term vol surface only, <1yr
        if tte < 1.0 and tte > 0.0: 
            call_chain, put_chain = spx.option_chain(expiry)
            if mode == 0:
                price = call_chain.apply(lambda row: 0.5 * (row['bid'] + row['ask']), axis=1)
            else:
                price = put_chain.apply(lambda row: 0.5 * (row['bid'] + row['ask']), axis=1)
            strikes = call_chain['strike']
            data = price.to_frame(name='price').join(strikes.to_frame(name='strike'))
            # Remove strikes which differ from underlying price outside range -50% -> 100% 
            mask = (underlying_price * 0.5 < data['strike']) & (underlying_price * 2. > data['strike'])
            data = data.loc[mask]
            data['underlying'] = np.repeat(underlying_price, data.shape[0])
            data['tenor'] = np.repeat(tte, data.shape[0])
            data = data.dropna() 
            data = data.reset_index(drop=True)
            # Get implied vol
            rate = rfr_func(tte)
            iv = data.apply(lambda row: find_iv(row, rate,mode), axis=1)
            tmp = np.vstack((data['tenor'].values, data['strike'].values, iv.values)).T
            # Remove 0 IV rows
            tmp = tmp[np.all(tmp > 0, axis=1)]
            dataset.append(tmp)

    dataset = np.concatenate(dataset)
    print(dataset.shape)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('Tenor (years)')
    ax.set_ylabel('Strike')
    ax.set_zlabel('IV')
    ax.scatter3D(dataset[:,0], dataset[:,1], dataset[:,2])
    plt.show()

#vol_surf("^SPX", mode=1)
vol_surf("AAPL", mode=1)


