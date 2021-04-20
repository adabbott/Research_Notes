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
import peslearn


from black_scholes import implied_vol
from risk_free_rate import get_rfr

def find_iv(row, rate):
    """
    Wrapper for pandas apply to find IV given market prices
    """
    try:
        result = implied_vol(row['price'], row['underlying'], row['strike'], row['tenor'], rate, mode=0) 
    except:
        result = 0.0
    return result

def vol_surf(ticker, mode=0):
    """
    Plots volatility surface using Black-Scholes
    ticker : string
        Ticker such as "^SPX", "AAPL", etc
    mode : int
        mode = 0 for calls, 1 for puts
    """
    # Get 15 min delayed SPX quote (delay matches option data.. i think)

    # Risk free rate as function of time in years.
    rfr_func = get_rfr()
    # Load ticker and expiries
    spx = yf.Ticker("^SPX")
    expiries = spx.options
    info = spx.info
    underlying_price = (info['bid'] + info['ask']) * 0.5

    # Will hold tenor, strike, IV pairs 
    dataset = []
    for expiry in expiries:
        # time til expiration
        tte = (datetime.strptime(expiry, "%Y-%m-%d") - datetime.now()) / timedelta(days=365)
        # Short term vol surface only, <1yr
        if tte < 1.0 and tte > 0.0: 
            call_chain, put_chain = spx.option_chain(expiry)
            price = call_chain.apply(lambda row: 0.5 * (row['bid'] + row['ask']), axis=1)
            strikes = call_chain['strike']
            data = price.to_frame(name='price').join(strikes.to_frame(name='strike'))
            data['underlying'] = np.repeat(underlying_price, data.shape[0])
            data['tenor'] = np.repeat(tte, data.shape[0])
            data = data.dropna() 
            data = data.reset_index(drop=True)
            # Get implied vol
            rate = rfr_func(tte)
            iv = data.apply(lambda row: find_iv(row, rate), axis=1)
            tmp = np.vstack((data['tenor'].values, data['strike'].values, iv.values)).T
            # Remove 0 IV rows
            tmp = tmp[np.all(tmp > 0, axis=1)]
            dataset.append(tmp)


    input_string = ("""
                   hp_maxit = 3
                   training_points = 700
                   rseed = 3
                   use_pips = false 
                   sampling = structure_based
                   """)

    gp = peslearn.ml.GaussianProcess(datasets[i], input_obj
    gp.optimize_model()
    errors.append(gp.test_error)


    dataset = np.concatenate(dataset)
    for i in dataset:
        print(i)
    #dataset = np.load('dataset.npy')

    #df = pd.DataFrame(dataset)
    #df.to_csv("dataset.csv", index=False)

    ## Interpolate for continuity with quick and dirty GP
    #kernel = RBF(2, ARD=True)  # TODO add HP control of kernel
    #model = GPRegression(dataset[:500,:2], dataset[:500,2].reshape(-1,1), kernel=kernel, normalizer=True)
    #model.optimize_restarts(10, optimizer="lbfgsb", robust=True, verbose=True, max_iters=1000, messages=True)
    
    #x = np.linspace(0.01,1,100)
    #y = np.linspace(3000, 5000, 100)
    #X, Y = np.meshgrid(x,y)
    #data_in = np.vstack((X.flatten(),Y.flatten())).T
    #print(data_in.shape)
    #for i in data_in:
    #    print(i)
    #print(X)
    #kprint(Y)
    #print(np.vstack((X.flatten(),Y.flatten())).shape)
    #print(np.vstack((X.flatten(),Y.flatten())))
    #p, cov = model.predict(data_in, full_cov=False)
    #print(p)

    # Create grid to evaluation model on and plot surface
    

    #fig = plt.figure()
    #ax = plt.axes(projection='3d')
    ##ax.scatter3D(dataset[:,0], dataset[:,1], dataset[:,2])
    #ax.plot_surface(dataset[:,0], dataset[:,1], dataset[:,2])
    #plt.show()


vol_surf("^SPX", mode=0)

#
#dataset = np.concatenate(dataset)
#print(dataset.shape)
#with open('dataset.npy', 'wb') as f:
#    np.save(f, dataset)
