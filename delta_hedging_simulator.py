import numpy as np

from pricer import OptionPricer
from financial_data import FinancialData, FinancialTools

class DeltaHedgingSimulator:
    def __init__(self, prices, K, T, r, sigma_imp, kp=0.0, kf=0.0):
            """
            prices : numpy array of underlying asset prices
            K : strike price
            T : maturity (years)
            r : riskless rate
            sigma_imp : implied volatility used for pricing and delta computation
            kp : proportional fees
            kf : fixed fees
            """
            self.prices = np.asarray(prices)
            self.N = len(self.prices)
            self.K = K
            self.T = T
            self.r = r
            self.sigma_imp = sigma_imp
            self.kp = kp
            self.kf = kf

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

            # Price the option and compute delta
            pricer = OptionPricer(self.prices[k], self.K, self.r, self.sigma_imp, tau, option_type="call", method="bsm")
            calls[k] = pricer.price()
            deltas[k] = pricer.greeks()['delta']

            if k == 0:
                # Initial setup: we sell the option and hedge with delta shares
                cash[k] = calls[k] - self.kf - self.kp * deltas[k] * self.prices[k] - deltas[k] * self.prices[k]
                portfolio_value[k] = -calls[k] + deltas[k] * self.prices[k] + cash[k]
                costs[k] = self.kf + self.kp * deltas[k] * self.prices[k]
            else:
                # interests earned on riskless cash position
                cash_with_interests = cash[k - 1] * np.exp(self.r * dt)

                # Rebalance the portfolio
                delta_change = deltas[k] - deltas[k-1]

                # If the change in delta is too small, we consider no transaction (so no transaction fees)
                if delta_change < 1e-10:
                     fixed_fee = 0
                else:
                     fixed_fee = self.kf
                
                cash[k] = cash_with_interests - delta_change * self.prices[k] - fixed_fee - self.kp * abs(delta_change) * self.prices[k]
                portfolio_value[k] = -calls[k] + deltas[k] * self.prices[k] + cash[k]
                costs[k] = costs[k-1] + fixed_fee + self.kp * abs(delta_change) * self.prices[k]

        return {
            "portfolio_value": portfolio_value,
            "cash": cash,
            "deltas": deltas,
            "calls": calls,
            "costs": costs
        }

    