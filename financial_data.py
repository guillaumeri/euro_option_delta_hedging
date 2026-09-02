import numpy as np
import yfinance as yf
import pandas as pd

class FinancialData:

    def __init__(self, data_type="Real"):
        self.data_type = data_type # "Real" or "Simulated"

    def get_data_real(self, ticker="^GSPC", period="10y"):
        """
        Gets historical data for the given ticker and period using yfinance.
        """
        self.ticker = ticker
        self.period = period
        df = yf.Ticker(self.ticker).history(period=self.period)
        self.data = df
        self.dates = df.index
        self.prices = df["Close"].to_numpy().flatten()
        return self.prices

    def get_data_simulated(self, S0=100, mu=0.05, sigma=0.2, T=1, N=252):
        """
        Simulates stock price data using Geometric Brownian Motion.
        """
        np.random.seed(1)
        dt = T/N 
        t = np.linspace(0, T, N+1)
        dW = np.random.standard_normal(N) * np.sqrt(dt)
        Wt = np.zeros(N+1) # N+1 to have W0 = 0
        Wt[1:] = np.cumsum(dW) # standard brownian motion
        X = (mu - 0.5 * sigma**2) * t + sigma * Wt # Using Ito's lemma on X = ln(S) with dS = mu*S*dt + sigma*S*dW, & integrating
        S = S0 * np.exp(X) # geometric brownian motion
        self.data = pd.DataFrame(data=S, index=pd.date_range(start=pd.Timestamp.today(), periods=N+1), columns=["Close"])
        self.dates = self.data.index
        self.prices = self.data["Close"].to_numpy().flatten()
        return self.prices

    def get_data(self, **kwargs):
        """Gets the data based on the data_type specified during initialization."""
        if self.data_type == "Real":
            return self.get_data_real(**kwargs)
        elif self.data_type == "Simulated":
            return self.get_data_simulated(**kwargs)

class FinancialTools:
    """A class containing financial tools for data analysis."""
    def daily_returns(self,prices):
        """Computes the daily returns"""
        return (prices[1:] - prices[:-1]) / prices[:-1]
            
    def daily_volatility(self,returns):
        """Computes the volatility (daily)."""
        return np.std(returns, ddof=1)

    def sharpe_ratio(self,returns, duration=252):
        """Computes the sharpe ratio for a given duration, default is annualized (252 days)."""
        vol = self.daily_volatility(returns)
        if vol == 0:
            return 0.0
        return (np.mean(returns) / vol) * np.sqrt(duration)

    