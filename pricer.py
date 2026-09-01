import numpy as np
import scipy.stats as sp


class OptionPricer:
    def __init__(self,
        current_stock_price: float,
        strike_price: float,
        risk_free_rate: float,
        volatility: float,
        T: float,
        option_type: str = "call",
        method: str = "bsm", 
        N: int = 100):

        self.current_stock_price = current_stock_price
        self.strike_price = strike_price
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.T = T
        self.option_type = option_type # "call" or "put"
        self.method = method # "bsm" or "binomial"
        self.N = N

    def price_BSM(self):
        d1 = (np.log(self.current_stock_price / self.strike_price) + (self.risk_free_rate + 0.5 * self.volatility**2) * self.T) / (self.volatility * np.sqrt(self.T))
        d2 = d1 - self.volatility * np.sqrt(self.T)

        if self.option_type == "call":
            return self.current_stock_price * sp.stats.norm.cdf(d1) - self.strike_price * np.exp(-self.risk_free_rate * self.T) * sp.stats.norm.cdf(d2)
        elif self.option_type == "put":
            return -self.current_stock_price * sp.stats.norm.cdf(-d1) + self.strike_price * np.exp(-self.risk_free_rate * self.T) * sp.stats.norm.cdf(-d2)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
    
    def price_binomial(self):

        current_stock_price = np.atleast_1d(self.current_stock_price)
        strike_price = np.atleast_1d(self.strike_price)
        dt = self.T/self.N
        up = np.exp(self.volatility*np.sqrt(dt))
        down = 1/up
        risk_free_proba = (np.exp(self.risk_free_rate*dt) - down)/(up - down)
        discount = np.exp(-self.risk_free_rate*dt)

        # Vectorization : current_stock_price and strike_price can be np arrays, so we need to add dimensions for each of them.
        # These vectors will have shape [amount of stock prices, amount of strike prices, inner indexing (all possible paths in the binomial tree)]
        j = np.arange(self.N+1)
        current_stock_price_3d, strike_price_3d, j_3d = np.meshgrid(current_stock_price, strike_price, j, indexing='ij') 
        
        # Stock value at maturity
        mature_stock_price_3d = current_stock_price_3d * (up**j_3d) * (down**(self.N-j_3d))

        # Option payoff at maturity
        if self.option_type == "call":
            option_payoff = np.maximum(mature_stock_price_3d - strike_price_3d,0)
        elif self.option_type == "put": 
            option_payoff = np.maximum(strike_price_3d - mature_stock_price_3d,0)
        else :
            raise ValueError("option_type must be 'call' or 'put'")

        # Backpropagation to fill the binomial tree until the root
        for i in range(self.N - 1, -1, -1):
            option_payoff = discount * (risk_free_proba * option_payoff[...,1:] + (1 - risk_free_proba) * option_payoff[...,:-1])

        return np.squeeze(option_payoff) # squeeze to avoid [[[result]]]

    def price(self):
        if self.method == "bsm":
            return self.price_BSM()
        elif self.method == "binomial":
            return self.price_binomial()
        else:
            raise ValueError(f"Unknown pricing method: {self.method}")