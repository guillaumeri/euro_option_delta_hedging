# European Option Delta-Hedging Analysis

A quantitative framework analyzing the risks and trade-offs of European Option Delta-Hedging with transaction fees and volatility misestimations.

**Note**: This repository is designed to be read alongside the main analysis notebook. Please refer directly to:
**[`notebooks/01_delta_hedging_analysis.ipynb`](notebooks/01_delta_hedging_analysis.ipynb)**

---

## Core Objectives

* **Monte Carlo delta-hedging simulation**: For short Call positions under BSM assumptions. It handles vectorized operation for performance as well as real or synthetic financial data.
* **Empirical Backtesting**: Evaluating delta-hedging strategy on real historical S&P 500 data.
* **Sensitivity analysis**: Quantifying the trade-off between portfolio readjustment frequency and volatility miscalibration with the **Sharpe Ratio** of the hedging strategy.

---
## Main files
* **01_delta_hedging_analysis.ipynb** : Main notebook, analysis and plots included.
* **financial_data.py** : Contains functions to generate data with desired characteristics (geometric brownian motion, can generate N paths at the same time in a vectorized manner using numpy), as well as fetching historical financial data.
* **pricer.py** : Contains functions that aim at pricing an option. Can handle two different methods (Black-Scholes-Merton model or Cox-Ross-Rubinstein binomial model).
* **delta_hedging_simulator.py** : Contains functions used to simulate delta-hedging on an underlying asset.

---

## Repository Structure

```text
.
├── notebooks/
│   └── 01_delta_hedging_analysis.ipynb  # Primary notebook (Full Analysis & Plots)
├── src/
│   └── financial_data.py
|   └── pricer.py               
│   └── delta_hedging_simulator.py           
├── requirements.txt
└── README.md
```

---
## Sources

* Hilpisch, Y. (2015). Derivatives analytics with Python: Data analysis, models, simulation, calibration and hedging. John Wiley & Sons.
* Hull, J. C. (2022). Options, futures, and other derivatives (11th ed., Global ed.). Pearson.
* Wilmott, P. (2007). Paul Wilmott introduces quantitative finance (2nd ed.). John Wiley & Sons.
