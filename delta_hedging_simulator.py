import numpy as np
import pandas as pd
from pricer import OptionPricer
from financial_data import FinancialData, FinancialTools

class DeltaHedgingSimulator:
    """A class to simulate delta hedging strategy for options, as a market-maker."""
    def __init__(self, prices, K, T, r, sigma_imp, kp=0.0, kf=0.0, n_options=100):
            """
            prices : numpy array of underlying asset prices
            K : strike price
            T : maturity (years)
            r : riskless rate
            sigma_imp : implied volatility used for pricing and delta computation
            kp : proportional fees
            kf : fixed fees
            n_options : number of options sold (default is 100)
            """
            self.prices = np.asarray(prices)
            self.N = len(self.prices)
            self.K = K
            self.T = T
            self.r = r
            self.sigma_imp = sigma_imp
            self.kp = kp
            self.kf = kf
            self.n_options = n_options
    def simulation(self):
        """Simulates the delta hedging strategy."""

        t = np.linspace(0, self.T, self.N)
        if self.N > 1:
            dt = t[1] - t[0]
        else:
            dt = self.T

        # Initialize arrays for each time step
        portfolio_value = np.zeros(self.N)
        cash = np.zeros(self.N)
        deltas = np.zeros(self.N)
        calls = np.zeros(self.N)
        costs = np.zeros(self.N)

        for k in range(self.N):
            # Time to maturity
            tau = self.T - t[k] 



            if abs(tau) < 1e-10: # to avoid division by zero in the pricer
                calls[k] = max(0.0, self.prices[k] - self.K)
                if self.prices[k] > self.K:
                    deltas[k] = 1.0 # option in the money : we hold 1 share of the underlying 
                else:
                    deltas[k] = 0.0 # option out of the money : we hold 0 share of the underlying
            else:
                # Price the option and compute delta
                pricer = OptionPricer(self.prices[k], self.K, self.r, self.sigma_imp, tau, option_type="call", method="bsm")
                calls[k] = pricer.price()
                deltas[k] = pricer.greeks()['delta']

            if k == 0:
                # Initial setup: we sell the option and hedge with delta shares

                if abs(deltas[k]) < 1e-10:
                    fixed_fee = 0
                    prop_fee = 0
                else:
                    fixed_fee = self.kf
                    prop_fee = self.n_options*self.kp * abs(deltas[k]) * self.prices[k]

                cash[k] = self.n_options*calls[k] - fixed_fee - prop_fee - self.n_options*deltas[k] * self.prices[k]
                portfolio_value[k] = -self.n_options*calls[k] + self.n_options*deltas[k] * self.prices[k] + cash[k]
                costs[k] = fixed_fee + prop_fee
            else:
                # interests earned on riskless cash position
                cash_with_interests = cash[k - 1] * np.exp(self.r * dt)

                # Rebalance the portfolio
                delta_change = deltas[k] - deltas[k-1]

                # If the change in delta is too small, we consider no transaction (so no transaction fees)
                if abs(delta_change) < 1e-10:
                     fixed_fee = 0
                     prop_fee = 0
                else:
                     fixed_fee = self.kf
                     prop_fee = self.n_options * self.kp * abs(delta_change) * self.prices[k]
                
                cash[k] = cash_with_interests - self.n_options * delta_change * self.prices[k] - fixed_fee - prop_fee
                portfolio_value[k] = -self.n_options * calls[k] + self.n_options * deltas[k] * self.prices[k] + cash[k]
                costs[k] = costs[k-1] + fixed_fee + prop_fee

        return pd.DataFrame(
            {
                "price": self.prices,
                "call": calls,
                "delta": deltas,
                "cash": cash,
                "costs": costs,
                "portfolio_value": portfolio_value,
            },
            index=t,
        )



    