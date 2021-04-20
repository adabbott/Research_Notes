import requests
import numpy as np
from bs4 import BeautifulSoup
from scipy.interpolate import interp1d

def get_rfr():
    """
    Scrapes current yield curve and 
    returns interpolation function to compute 
    risk-free rate given amount of time in years.
    """
    r = requests.get("http://www.treasury.gov/resource-center/data-chart-center/interest-rates/Pages/TextView.aspx?data=yield")
    soup = BeautifulSoup(r.text, 'html.parser')
    # Table and rows represented in html by `<table class="t-chart"` and `tr class`, respectively
    table = soup.find("table", attrs={'class', 't-chart'})
    rows = table.find_all('tr')
    # Get most recent rates 
    current_rates = rows[len(rows)-1].find_all("td")
    date = current_rates[0].get_text()
    # pull out each rate, convert from basis points to % 
    # This breaks if rates go neg
    rates = [0.0] + [float(current_rates[i].get_text()) * 0.01 for i in range(1,13)]
    # 0m, 1m, 2m, 3m, 6m, 1y, 2y, 3y, 5y, 7y, 10y, 20y, 30y
    years = [0.0, 1/12, 2/12, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
    f = interp1d(years, rates)
    return f

