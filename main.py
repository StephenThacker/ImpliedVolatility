import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import plotly.graph_objects as go
import datetime as dt
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from massive import RESTClient
from dotenv import load_dotenv
import os
import time
from numba import njit, prange



class binomial_tree_vectorized():

    def __init__(self, strike_price, number_of_layers, initial_stock_price, interest_rate, time_to_expiration, stock_dividend,call_or_put):
        self.strike_price = strike_price
        self.number_of_layers = number_of_layers
        self.initial_stock_price = initial_stock_price
        self.time_to_expiration = time_to_expiration
        self.interest_rate = interest_rate
        self.dividend = stock_dividend
        self.node_list = []
        self.call_or_put = call_or_put
        self.time_to_expiration = self.time_to_expiration/365
        try:
            self.delta_t = self.time_to_expiration / (self.number_of_layers -1)
        except ZeroDivisionError:
            raise ValueError

    @staticmethod
    @njit(fastmath = True)
    def forward_pass_njit(number_of_layers,initial_stock_price,down_factor,up_factor):
        price_array = np.zeros((number_of_layers,number_of_layers))
        price_array[0,0] = initial_stock_price
        for i in range(1,number_of_layers):
            price_array[i,0] = price_array[i-1,0]*down_factor
            price_array[i,1:i+1] = price_array[i-1,0:i]*up_factor
        return price_array
    
    @staticmethod
    @njit(fastmath = True)
    def backwards_pass_njit(price_array,number_of_layers,discount_up,discount_down,strike, call_or_put):
        options_array  = np.zeros((number_of_layers,number_of_layers))
        if call_or_put == True:
            options_array[-1,:] = np.maximum(price_array[-1,:] - strike, 0)
        if call_or_put == False:
            options_array[-1,:] = np.maximum(strike - price_array[-1,:], 0)
        for i in range(number_of_layers -2, -1,-1):
            continuation = discount_up*options_array[i+1,1:i+2] + discount_down*options_array[i+1,0:i+1]
            intrinsic = np.maximum(price_array[i,0:i+1] - strike,0) if call_or_put == True else np.maximum(strike - price_array[i,0:i+1],0)
            options_array[i,0:i+1] = np.maximum(continuation,intrinsic)

        return options_array[0,0]
   
    #builds out the tree
    #Uses 2d numpy array to create pricing array
    def pricing_forward_pass(self,sigma, strike):
        call_or_put = self.call_or_put
        up_factor, down_factor = self.define_time_segment(sigma)
        number_of_layers = self.number_of_layers
        prob = self.calculate_probability(up_factor,down_factor)
        risk_free = self.interest_rate
        discount = np.exp(-1*risk_free*self.delta_t)
        discount_up = discount*prob
        discount_down = discount*(1-prob)
        initial_stock_price = self.initial_stock_price
        if call_or_put == "call":
            call_or_put = True
        else:
            call_or_put = False
        
        price_array = self.forward_pass_njit(number_of_layers,initial_stock_price,down_factor,up_factor)

        
        return self.backwards_pass_njit(price_array,number_of_layers,discount_up,discount_down,strike, call_or_put)


        
    def vectorization_of_forward_pass(self,sigma, strike):
        return self.pricing_forward_pass(sigma, strike )
    
    def define_time_segment(self,sigma):
       
        u = np.exp(sigma * np.sqrt(self.delta_t))
        d = np.exp(-1*sigma * np.sqrt(self.delta_t))
        return [u,d]
    
    def calculate_probability(self,u,d):
        try:
            return (np.exp((self.interest_rate-self.dividend) * self.delta_t) - d) / (u - d)
        except ZeroDivisionError:
            raise ValueError("Division by zero in probability calculation (u == d).")
    



class options_chain:
    
    def __init__(self, ticker, method,number_of_layers, steps, call_or_put):
        self.ticker = ticker
        self.options_chains_dict = {}
        self.method = method
        self.number_of_layers = number_of_layers
        self.call_or_put = call_or_put
        self.fetch_options_data(self.ticker)

    def vega(self, stock_price, time_to_expiration, cont_div_yield, d_1):
        vega = stock_price*np.exp(-1*cont_div_yield*time_to_expiration)*1/np.sqrt(2*np.pi)*np.exp(-1*(d_1**2)/2)*np.sqrt(time_to_expiration)
        return vega

    def black_scholes_call_option(self, stock_price, contin_div_yield, time_to_expiration,d_1, d_2, strike_price , risk_free_rate):
        price = stock_price*np.exp(-contin_div_yield*time_to_expiration)*norm.cdf(d_1) - strike_price*np.exp(-risk_free_rate*time_to_expiration)*norm.cdf(d_2)
        return price
    
    def black_scholes_put_option(self, stock_price, contin_div_yield, time_to_expiration, d_1, d_2, strike_price, risk_free_rate):
        price = strike_price*np.exp(-1*risk_free_rate*time_to_expiration)*norm.cdf(-1*d_2) - stock_price*np.exp(-1*contin_div_yield*time_to_expiration)*norm.cdf(-1*d_1)
        return price

    def d_1(self, strike_price, stock_price, risk_free_rate, cont_div_yield, sigma, time_to_expiration):
        d_1 = (np.log(stock_price/strike_price)+(risk_free_rate - cont_div_yield + (sigma**2)/2)*time_to_expiration)/(sigma*np.sqrt(time_to_expiration))
        return d_1

    def d_2(self, d_1, time_to_expiration, sigma):
        d2 = d_1 - sigma*np.sqrt(time_to_expiration)
        return d2
    
    def newton_raphson_method_black_scholes(self, epsilon,stock_price, option_price,risk_free_rate, contin_div_yield, time_to_expiration, strike_price, bs_price_func):
        sigma = 0.8
        black_scholes_cost = 1
        market_cost = 0
        max_iter = 400
        count = 0
        bs_price_func = bs_price_func
        while (np.abs(black_scholes_cost - market_cost) > epsilon) and count< max_iter:

            d_1 = self.d_1(strike_price, stock_price, risk_free_rate,contin_div_yield,sigma,time_to_expiration)
            d_2 = self.d_2(d_1, time_to_expiration, sigma)
            black_scholes_cost = bs_price_func(stock_price, contin_div_yield, time_to_expiration, d_1, d_2, strike_price, risk_free_rate)
            market_cost = option_price
            vega = self.vega(stock_price, time_to_expiration, contin_div_yield, d_1)

            if abs(vega) < 1e-8:
                break
            sigma = sigma - (black_scholes_cost - market_cost)/(vega)

            count += 1
        return sigma
    
    def dates_to_expiration_fraction(self, expiration_date):
        expiration_date = dt.datetime.strptime(expiration_date, "%Y-%m-%d").date()
        days_to_expiration = (expiration_date - dt.date.today()).days
        fraction_of_days = days_to_expiration/365
        return fraction_of_days
    
    def dates_to_expiration_days(self,expiration_date):
        expiration_date = dt.datetime.strptime(expiration_date, "%Y-%m-%d").date()
        days_to_expiration = (expiration_date - dt.date.today()).days
        return days_to_expiration
    
    def num_dates_to_expir(self,expiration_date):
        expiration_date = dt.datetime.strptime(expiration_date, "%Y-%m-%d").date()
        days_to_expiration = (expiration_date - dt.date.today()).days

        return days_to_expiration
    
    def calc_implied_volatility(self,number_of_layers):
        if self.method == 'Black Scholes':
            if self.call_or_put == "call":
                bs_price_func = self.black_scholes_call_option
            if self.call_or_put == "put":
                bs_price_func = self.black_scholes_put_option
            for key in self.options_chains_dict.keys():
                date_fraction = self.dates_to_expiration_fraction(key)
                self.options_chains_dict[key]['calcImpliedVol'] = np.vectorize(self.newton_raphson_method_black_scholes)(1e-5, self.last_stock_price,\
                                                                                                                         self.options_chains_dict[key]['midpoint'],\
                                                                                                                            self.risk_free,\
                                                                                                                                self.dividend_yield,\
                                                                                                                                    date_fraction,\
                                                                                                                                        self.options_chains_dict[key]['strike'],\
                                                                                                                                            bs_price_func)
                self.options_chains_dict[key]['daystoExpir'] = np.vectorize(self.num_dates_to_expir)(key)

        start_time = time.perf_counter()
        if self.method == "BinTree Continuous Deriv":
                count = 0
                for key in self.options_chains_dict.keys():
                    dates_to_expiration = self.dates_to_expiration_days(key)
                    self.options_chains_dict[key]['daystoExpir'] = np.vectorize(self.num_dates_to_expir)(key)
                    self.tree = binomial_tree_vectorized(self.options_chains_dict[key]['strike'],\
                                             self.number_of_layers,self.last_stock_price,\
                                                self.risk_free, dates_to_expiration,\
                                                    self.dividend_yield,\
                                                        self.call_or_put)
                    self.options_chains_dict[key]['calcImpliedVol'] = np.vectorize(self.vectorized_brentq_wrapper)(0.01,2, self.options_chains_dict[key]['strike'],\
                                                                                                                  self.options_chains_dict[key]['midpoint'])
  
        end_time = time.perf_counter()
        print("final time", end_time - start_time)      
        return
    
   
    def vectorized_brentq_wrapper(self,sigma_low,sigma_high,strike_price,midpoint, xtol=1e-8, rtol=1e-8, maxiter=100):
        def brentq_objective(sigma):
            return self.tree.vectorization_of_forward_pass(sigma,strike_price) - midpoint

        try:
            start = time.perf_counter()
            result = brentq(brentq_objective, sigma_low, sigma_high, xtol=1e-8, rtol=1e-8, maxiter=100)
            stop = time.perf_counter()
            print("time brentq",stop-start)
            return result
        except ValueError:
            return np.nan
    
    def fetch_options_data(self, ticker):
        self.ticker = yf.Ticker(ticker)
        expiration_dates = self.ticker.options
        self.info = self.ticker.info
        if self.call_or_put == "call":
            self.options_chains_dict = {exp : self.ticker.option_chain(exp).calls for exp in expiration_dates}
        if self.call_or_put =="put":
            self.options_chains_dict = {exp : self.ticker.option_chain(exp).puts for exp in expiration_dates}

        for exp, df in self.options_chains_dict.items():
            if 'bid' in df.columns and 'ask' in df.columns:
                df['midpoint'] = (df['bid'] + df['ask']) / 2
        self.last_stock_price = self.ticker.history(period="1d")['Close'].iloc[-1]
        self.risk_free = yf.Ticker("^IRX").history(period="1d")['Close'][-1]/100
        if yf.Ticker(ticker).info.get('dividendYield') != None:
            self.dividend_yield = yf.Ticker(ticker).info.get('dividendYield')/100
        else:
            self.dividend_yield = 0

        self.calc_implied_volatility(self.number_of_layers)


    def plot_imp_vol_surface(self):
        last_price = self.last_stock_price

        lower_strike = last_price * 0.7
        upper_strike = last_price * 1.3

        max_days_to_exp = 150

        strikes_all = []
        maturities_all = []
        implied_vols_all = []

        for exp_date, df in self.options_chains_dict.items():
            days_to_exp = self.num_dates_to_expir(exp_date)
            if days_to_exp > max_days_to_exp or days_to_exp < 0:
                continue
            #purge negative implied vol, excessively high implied vol, deep OTM or ITM options by strike,
            valid_mask = (~df['calcImpliedVol'].isna()) & \
                        (df['calcImpliedVol'] > 0) & \
                        (df['calcImpliedVol'] <= 2.0) & \
                        (df['strike'] >= lower_strike) &\
                              (df['strike'] <= upper_strike) &\
                              (df['bid']>0) & (df['ask'] >0) &\
                              (df['volume']>0)            

            strikes = df.loc[valid_mask, 'strike'].values
            maturity = days_to_exp
            implied_vols = df.loc[valid_mask, 'calcImpliedVol'].values

            if len(strikes) == 0:
                continue

            strikes_all.extend(strikes)
            maturities_all.extend([maturity] * len(strikes))
            implied_vols_all.extend(implied_vols)

        strikes_all = np.array(strikes_all)
        maturities_all = np.array(maturities_all)
        implied_vols_all = np.array(implied_vols_all)

        

        if len(strikes_all) == 0:
            print("No data points available for plotting after filtering.")
            return

        strike_grid = np.linspace(strikes_all.min(), strikes_all.max(), 50)
        maturity_grid = np.linspace(maturities_all.min(), maturities_all.max(), 50)
        M_grid, K_grid = np.meshgrid(maturity_grid, strike_grid)

        IV_grid = griddata(
            points=(maturities_all, strikes_all),
            values=implied_vols_all,
            xi=(M_grid, K_grid),
            method='linear'
        )

        fig = go.Figure(data=[go.Surface(
            x=M_grid,
            y=K_grid,
            z=IV_grid,
            colorscale='Viridis',
            colorbar=dict(title='Implied Volatility')
        )])

        fig.update_layout(
            title=f"Implied Volatility Surface for {self.ticker.ticker} (Filtered)",
            scene=dict(
                xaxis_title='Days to Expiration',
                yaxis_title='Strike Price',
                zaxis_title='Implied Volatility',
            ),
            autosize=True,
            width=800,
            height=700
        )

        fig.show()

def testing_polygon_api(ticker,client):
    now = dt.datetime.utcnow()
    thirty_min_ago = now - dt.timedelta(minutes=30)

    from_date = thirty_min_ago.strftime("%Y-%m-%d")
    to_date = now.strftime("%Y-%m-%d")

    '''aggs = client.list_aggs(
    ticker=ticker,
    multiplier=1,          # 1-minute bars
    timespan="minute",     # aggregate by minute
    from_=from_date,
    to=to_date,
    adjusted=True,
    limit=10)'''               # get up to 50 bars)

    for contract in client.list_options_contracts("AAPL"):
        print(contract.ticker)
    
    return


def main():
    #load_dotenv()
    #api_key0 = os.getenv("API_KEY")
    #client = RESTClient(api_key = api_key0)
    #contracts = client.list_options_contracts(underlying_ticker="AAPL", limit=100)

    #testing_polygon_api("AAPL",client)
    call_bin_options = options_chain("BBWI", "BinTree Continuous Deriv", 100,30, "call")
    call_put_options = options_chain("BBWI","BinTree Continuous Deriv", 100,30, "put" )
    call_options_scholes = options_chain("BBWI", "Black Scholes",None,None,"call")
    put_options_scholes = options_chain("BBWI", "Black Scholes", None, None, "put")

    call_bin_options.plot_imp_vol_surface()
    call_options_scholes.plot_imp_vol_surface()
    call_put_options.plot_imp_vol_surface()
    put_options_scholes.plot_imp_vol_surface()

if __name__ == "__main__":
    main()
