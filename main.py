import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
import datetime as dt

class options_chain:
    
    def __init__(self, ticker):
        self.ticker = ticker
        self.fetch_options_data(self.ticker)

    def vega(self, stock_price, time_to_expiration, cont_div_yield, d_1):
        vega = stock_price*np.exp(-1*cont_div_yield*time_to_expiration)*1/np.sqrt(2*np.pi)*np.exp(-1*(d_1**2)/2)*np.sqrt(time_to_expiration)
        return vega

    def black_scholes_call_option(self, stock_price, contin_div_yield, time_to_expiration,d_1, d_2, strike_price , risk_free_rate):
        price = stock_price*np.exp(-contin_div_yield*time_to_expiration)*norm.cdf(d_1) - strike_price*np.exp(-risk_free_rate*time_to_expiration)*norm.cdf(d_2)
        return price

    def d_1(self, strike_price, stock_price, risk_free_rate, cont_div_yield, sigma, time_to_expiration):
        d_1 = (np.log(stock_price/strike_price)+(risk_free_rate - cont_div_yield + (sigma**2)/2)*time_to_expiration)/(sigma*np.sqrt(time_to_expiration))
        return d_1

    def d_2(self, d_1, time_to_expiration, sigma):
        d2 = d_1 - sigma*np.sqrt(time_to_expiration)
        return d2
    
    def newton_raphson_method(self, epsilon,stock_price, option_price,risk_free_rate, contin_div_yield, time_to_expiration, strike_price):
        sigma = 0.2
        black_scholes_cost = 1
        market_cost = 0
        max_iter = 400
        count = 0
        while (np.abs(black_scholes_cost - market_cost) > epsilon) and count< max_iter:

            d_1 = self.d_1(strike_price, stock_price, risk_free_rate,contin_div_yield,sigma,time_to_expiration)
            d_2 = self.d_2(d_1, time_to_expiration, sigma)
            black_scholes_cost = self.black_scholes_call_option(stock_price, contin_div_yield, time_to_expiration, d_1, d_2, strike_price, risk_free_rate)
            market_cost = option_price
            vega = self.vega(stock_price, time_to_expiration, contin_div_yield, d_1)

            if vega < 1e-8:
                print("here")
                break
            sigma = sigma - (black_scholes_cost - market_cost)/(vega)

            count += 1
        return sigma

    def fetch_options_data(self, ticker):
        ticker = yf.Ticker(ticker)
        expiration_dates = ticker.options
        print(expiration_dates)
        options = ticker.option_chain(expiration_dates[2])
        calls = options.calls
        selected_row = calls[calls['strike'] == 190]
        expiry_date = dt.datetime.strptime(expiration_dates[2], "%Y-%m-%d").date()
        today = dt.date.today()
        days_to_expiration = (expiry_date - today).days
        time_to_expiration = days_to_expiration / 365        
        info = ticker.info
        last_stock_price = info.get('regularMarketPrice') or ticker.history(period="1d")['Close'][0]
        risk_free = yf.Ticker("^IRX").history(period="1d")['Close'][-1]/100
        dividend_yield = yf.Ticker("NVDA").info.get('dividendYield')/100 or 0
        print("implied volatility")
        print(selected_row['impliedVolatility'])
        calc_implied_vol = self.newton_raphson_method(1e-5, last_stock_price, selected_row['lastPrice'].item(),risk_free,dividend_yield,time_to_expiration,190 )
        print("calculating implied vol")
        print(calc_implied_vol)


        return


def main():
    nvda_options = options_chain("NVDA")

if __name__ == "__main__":
    main()
