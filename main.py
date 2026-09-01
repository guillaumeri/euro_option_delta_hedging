import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import yfinance as yf

import pricer


S0 = 100
K = 100
T = 1
r = 5/100
sigma = 0.1

print(pricer.OptionPricer(S0, K, r, sigma, T, option_type="call", method="bsm").price())
print(pricer.OptionPricer(S0, K, r, sigma, T, option_type="call", method="bsm").greeks()['delta'])
