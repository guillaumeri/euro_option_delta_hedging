import numpy as np
import yfinance as yf
import pandas as pd

class FinancialData:

    def __init__(self, ticker="^GSPC", period="10y"):
        self.ticker = ticker
        self.period = period
        # We load the data right away
        self.get_data()

    def get_data(self):
        """
        Gets historical data for the given ticker and period using yfinance.
        """
        df = yf.Ticker(self.ticker).history(period=self.period)
        self.data = df
        self.dates = df.index
        self.prices = df["Close"].to_numpy().flatten()

class FinancialTools:
    """A class containing financial tools for data analysis."""
    def daily_returns(self,prices):
        """Computes the daily returns.
        Input : prices (numpy array, same size as prices)"""
        return (prices[1:] - prices[:-1]) / prices[:-1]
            
    def daily_volatility(self,returns):
        """Computes the volatility (daily).
        Input : returns (numpy array of strategy/asset returns)"""
        return np.std(returns, ddof=1)

    def sharpe_ratio(self,returns, duration=252):
        """Computes the sharpe ratio for a given duration, default is annualized (252 days).
        Input : returns (numpy array of strategy/asset returns), duration (optional, int)"""
        vol = self.daily_volatility(returns)
        if vol == 0:
            return 0.0
        return (np.mean(returns) / vol) * np.sqrt(duration)

    