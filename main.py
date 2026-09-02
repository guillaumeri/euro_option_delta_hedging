import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import yfinance as yf
import seaborn as sns

from pricer import OptionPricer
from financial_data import FinancialData, FinancialTools
from delta_hedging_simulator import DeltaHedgingSimulator

## Delta-Hedging : known volatility, known riskless rate
# Simulation parameters
S0 = 100.0  # initial stock prize
K = 100.0  # strike prize
T = 1.0  # time to maturity (in year)
r = 0.03  # riskless rate
sigma_real = 0.20  # Actual volatility of underlying
sigma_imp = 0.20  # implicit volatility (for pricing and delta)
N = 252  # Amount of time steps : 1 year = 252 trading days
n_options = 100  # Amount of options sold
kp= 0.005 # proportional fees
kf= 0.1 # fixed fees


# Synthetic data generation
financial_data = FinancialData(data_type="Simulated")
prices = financial_data.get_data(S0=S0, mu=r, sigma=sigma_real, T=T, N=N)

# Delta hedging simulation
delta_hedging_simulator = DeltaHedgingSimulator(prices, K, T, r, sigma_imp, kp=kp, kf=kf, n_options=n_options)
df_results = delta_hedging_simulator.simulation()

# Plot : Decomposition of the portfolio value over time
plt.figure()
plt.plot(df_results.index, -n_options *df_results["call"], label="Value of call sold", color="red")
plt.plot(df_results.index, n_options *df_results["delta"] * df_results["price"], label="Hedge (underlying bought)", color="orange")
plt.plot(df_results.index, df_results["cash"], label="Cash", color="gray")
plt.plot(df_results.index,df_results["portfolio_value"],label="Net P&L (portfolio_value)",color="black",linewidth=3)
plt.plot(df_results.index, -df_results["costs"], label="Costs (fees)", color="purple")
plt.axhline(0, color="black", linestyle=":", alpha=0.6)
plt.title("Delta Hedging : Decomposition of the portfolio value over time")
plt.xlabel("Time (years)")
plt.ylabel("Value (€)")
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot : Evolution of Delta over time
plt.figure()
plt.plot(df_results.index, df_results["delta"], label="Delta", color="purple")
plt.title("Evolution of Delta over time")
plt.xlabel("Time (years)")
plt.ylabel("Delta")
plt.axhline(0, color="black", linestyle=":", alpha=0.6)
plt.legend(loc="upper left")
plt.show()


# Using different values of implied volatility to see the impact of mispricing on the P&L

kp= 0.0 # proportional fees (set to 0 to see the impact of mispricing only)
kf= 0.0 # fixed fees (set to 0 to see the impact of mispricing only)

sigma_imp_values = [0.15, 0.20, 0.25]  # Underestimated, correct, overestimated
plt.figure()
for sigma_imp in sigma_imp_values:
    delta_hedging_simulator = DeltaHedgingSimulator(prices, K, T, r, sigma_imp, kp=kp, kf=kf, n_options=n_options)
    df_results = delta_hedging_simulator.simulation()
    plt.plot(df_results.index, df_results["portfolio_value"], label=f"sigma_imp = {sigma_imp*100:.0f}%")
plt.axhline(0, color="black", linestyle=":", alpha=0.6)
plt.title("Impact of mispricing on P&L (real volatility : 20%)")
plt.xlabel("Time (years)")
plt.ylabel("P&L (€)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# Impact of the frequency of rebalancing on the P&L (with fees)

kp= 0.005 # proportional fees
kf= 0.1 # fixed fees

plt.figure()
for N_val in [12, 52, 252, 2520]:
    # Generate synthetic data for the given N_val
    p_sim = financial_data.get_data(S0=S0, mu=r, sigma=sigma_real, T=T, N=N_val)
    sim = DeltaHedgingSimulator(p_sim, K, T, r, sigma_imp=sigma_real, kp=kp, kf=kf)
    res = sim.simulation()
    plt.plot(res.index, res["portfolio_value"], label=f"N = {N_val} steps")

plt.axhline(0, color="black", linestyle=":", alpha=0.6)
plt.title("P&L according to the frequency N (with kp = 3 bps, kf = 0.01 €)")
plt.xlabel("Time (years)")
plt.ylabel("P&L (€)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

