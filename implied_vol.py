import yfinance as yf
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import plotly.graph_objects as go
import datetime as dt
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import os
import psycopg2
import time
from numba import njit, prange
import requests
import httpx
import io
from datetime import date, timedelta
import csv
import holidays

load_dotenv()

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
    
class calculate_dates:

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

    def check_if_day_is_trading_day(self, potential_day: dt ):

        if potential_day.weekday() >= 5:
            return False
        
        nyse_holidays = holidays.financial_holidays('NYSE')

        if potential_day in nyse_holidays:
            return False

        return True

class black_scholes_implied_volatility:

    def __init__(self):
        self.call_or_put_method = self.construct_call_or_put_dict()

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
    
    def construct_call_or_put_dict(self):
        func_dict= {'CALL' : self.black_scholes_call_option, 'PUT' : self.black_scholes_put_option}

        return func_dict
    
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




class y_finance_options_chain:
    
    def __init__(self, ticker, method,number_of_layers, steps, call_or_put):
        self.ticker = ticker
        self.options_chains_dict = {}
        self.method = method
        self.number_of_layers = number_of_layers
        self.call_or_put = call_or_put
        self.calculate_date_object = calculate_dates()
        self.black_scholes = black_scholes_implied_volatility()
        self.fetch_options_data(self.ticker)

    
    def calc_implied_volatility(self):
        if self.method == 'Black Scholes':
            if self.call_or_put == "call":
                bs_price_func = self.black_scholes.black_scholes_call_option
            if self.call_or_put == "put":
                bs_price_func = self.black_scholes.black_scholes_put_option
            for key in self.options_chains_dict.keys():
                date_fraction = self.calculate_date_object.dates_to_expiration_fraction(key)
                self.options_chains_dict[key]['calcImpliedVol'] = np.vectorize(self.black_scholes.newton_raphson_method_black_scholes)(1e-5, self.last_stock_price,\
                                                                                                                         self.options_chains_dict[key]['midpoint'],\
                                                                                                                            self.risk_free,\
                                                                                                                                self.dividend_yield,\
                                                                                                                                    date_fraction,\
                                                                                                                                        self.options_chains_dict[key]['strike'],\
                                                                                                                                            bs_price_func)
                self.options_chains_dict[key]['daystoExpir'] = np.vectorize(self.calculate_date_object.num_dates_to_expir)(key)
        #key is the contract expiration date
        #dates to expiration, is the dates the contract has to expire
        start_time = time.perf_counter()
        if self.method == "BinTree Continuous Deriv":
                for key in self.options_chains_dict.keys():
                    dates_to_expiration = self.calculate_date_object.dates_to_expiration_days(key)
                    self.options_chains_dict[key]['daystoExpir'] = np.vectorize(self.calculate_date_object.num_dates_to_expir)(key)
                    self.tree = binomial_tree_vectorized(self.options_chains_dict[key]['strike'],\
                                             self.number_of_layers,self.last_stock_price,\
                                                self.risk_free, dates_to_expiration,\
                                                    self.dividend_yield,\
                                                        self.call_or_put)
                                                        
                    vec_func = np.vectorize(self.vectorized_brentq_wrapper, otypes=[float])
                    self.options_chains_dict[key]['calcImpliedVol'] = vec_func(
                        0.01,
                        5,
                        self.options_chains_dict[key]['strike'].values,
                        self.options_chains_dict[key]['midpoint'].values
                    )
                    #self.options_chains_dict[key]['calcImpliedVol'] = np.vectorize(self.vectorized_brentq_wrapper)(0.01,2, self.options_chains_dict[key]['strike'],\
                                                                                                                  #self.options_chains_dict[key]['midpoint'])
  
        end_time = time.perf_counter()
        return
    
   
    def vectorized_brentq_wrapper(self,sigma_low,sigma_high,strike_price,midpoint, xtol=1e-8, rtol=1e-8, maxiter=100):
        def brentq_objective(sigma):
            return self.tree.vectorization_of_forward_pass(sigma,strike_price) - midpoint

        try:
            #start = time.perf_counter()
            result = brentq(brentq_objective, sigma_low, sigma_high, xtol=1e-8, rtol=1e-8, maxiter=100)
            #stop = time.perf_counter()
            #print("time brentq",stop-start)
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
            #print("df columns test")
            #print(df.columns)
            print(df.columns)
            if 'bid' in df.columns and 'ask' in df.columns:
                df['midpoint'] = (df['bid'] + df['ask']) / 2
            print(df.head())
        self.last_stock_price = self.ticker.history(period="1d")['Close'].iloc[-1]
        data = yf.Ticker("^IRX").history(period="5d")
        if not data.empty:
            self.risk_free = data['Close'].iloc[-1] / 100
        else:
            self.risk_free = 0.04


        if yf.Ticker(ticker).info.get('dividendYield') != None:
            self.dividend_yield = yf.Ticker(ticker).info.get('dividendYield')/100
        else:
            self.dividend_yield = 0

        self.calc_implied_volatility()



    def plot_imp_vol_surface(self):
        last_price = self.last_stock_price

        lower_strike = last_price * 0.7
        upper_strike = last_price * 1.5

        max_days_to_exp = 150

        log_moneyness_all = []
        maturities_all = []
        implied_vols_all = []

        for exp_date, df in self.options_chains_dict.items():
            days_to_exp = self.calculate_date_object.num_dates_to_expir(exp_date)
            if days_to_exp > max_days_to_exp or days_to_exp < 0:
                continue

            valid_mask = (~df['calcImpliedVol'].isna()) & \
                        (df['calcImpliedVol'] > 0) & \
                        (df['calcImpliedVol'] <= 2.0) & \
                        (df['strike'] >= lower_strike) & \
                        (df['strike'] <= upper_strike) & \
                        (df['bid'] > 0) & (df['ask'] > 0) & \
                        (df['volume'] > 0)

            strikes = df.loc[valid_mask, 'strike'].values
            maturity = days_to_exp
            implied_vols = df.loc[valid_mask, 'calcImpliedVol'].values

            if len(strikes) == 0:
                continue

            log_moneyness = np.log(strikes / last_price)

            log_moneyness_all.extend(log_moneyness)
            maturities_all.extend([maturity] * len(strikes))
            implied_vols_all.extend(implied_vols)

        log_moneyness_all = np.array(log_moneyness_all)
        maturities_all = np.array(maturities_all)
        implied_vols_all = np.array(implied_vols_all)

        if len(log_moneyness_all) == 0:
            print("No data points available for plotting after filtering.")
            return

        logm_grid = np.linspace(log_moneyness_all.min(), log_moneyness_all.max(), 50)
        maturity_grid = np.linspace(maturities_all.min(), maturities_all.max(), 50)

        M_grid, LM_grid = np.meshgrid(maturity_grid, logm_grid)

        IV_grid = griddata(
            points=(maturities_all, log_moneyness_all),
            values=implied_vols_all,
            xi=(M_grid, LM_grid),
            method='linear'
        )

        fig = go.Figure(data=[go.Surface(
            x=M_grid,
            y=LM_grid,
            z=IV_grid,
            colorscale='Viridis',
            colorbar=dict(title='Implied Volatility')
        )])

        fig.update_layout(
            title=f"Implied Vol Surface (Log-Moneyness) for {self.ticker.ticker}",
            scene=dict(
                xaxis_title='Days to Expiration',
                yaxis_title='Log-Moneyness ln(K/S)',
                zaxis_title='Implied Volatility',
            ),
            autosize=True,
            width=800,
            height=700
        )

        fig.show()


#target date is assummed to be a datetime object
class thetadata_options_scrape_EOD:
    def __init__(self, ticker,target_date):
        self.BASE_URL = "http://127.0.0.1:25503/v3"
        self.PARAMS = {}
        self.ticker = ticker
        self.target_date = target_date
        #might want to refactor calculate dates to be an instance passed to the class, to avoid redundant memory drag
        #If I'm going to pull data for like 500 tickers
        self.date_calculator = calculate_dates()
        self.black_scholes = black_scholes_implied_volatility()
        if self.date_calculator.check_if_day_is_trading_day(target_date) == False:
            raise ValueError(f"Target date {target_date} is a weekend or market holiday. Cannot scrape options data.")


    #for single ticker/expiration date, pulls all options data for a specific date and stores into database
    def options_api_pull_per_exp_date(self,ticker, target_date, expiration_date, conn_params):        
        #since expiration dates includes all expiration dates that have ever existed for options,
        # we need to filter dates that are not relevant on the target date.
        BASE_URL = "http://127.0.0.1:25503/v3"
        PARAMS = {'start_date': self.target_date,'end_date': self.target_date,'symbol': self.ticker,'expiration':expiration_date }

        start_date = dt.datetime.strftime(self.target_date,"%Y%m%d") 
        end_date = dt.datetime.strftime(self.target_date,"%Y%m%d") 

        PARAMS['start_date'] = start_date
        PARAMS['end_date'] = end_date

        url = BASE_URL + '/option/history/eod'

        with httpx.stream("GET", url, params = PARAMS, timeout=60) as response:
            response.raise_for_status()
            content = response.read()
            df = pd.read_csv(io.BytesIO(content), header = 0)

        df = df.rename(columns = {'right': 'option_type'})
        df['ticker'] = ticker
        df['price_date'] = target_date

        df['expiration'] = pd.to_datetime(df['expiration']).dt.date
        df['price_date'] = pd.to_datetime(df['price_date']).dt.date
        df['midpoint'] = (df['ask'] + df['bid'])/2
        cols = df.columns

        #purging any NaN values
        df = df[cols].where(df[cols].notnull(),None)

        cols = [ 'ticker', 'symbol', 'expiration','strike','option_type','created','price_date','last_trade',\
                'open', 'high','low', 'close', 'volume', 'count', 'bid_size','bid_exchange','bid','bid_condition',\
                    'ask_size','ask_exchange','ask','ask_condition', 'midpoint']
        
        insert_sql = '''INSERT INTO options (
        ticker, symbol, expiration, strike,option_type,created,price_date,last_trade,\
                open, high,low, close, volume, count, bid_size,bid_exchange,bid,bid_condition,\
                    ask_size,ask_exchange,ask,ask_condition, midpoint )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING;
      
        '''

        try:
            with psycopg2.connect(**conn_params) as conn:
                with conn.cursor() as cur:
                    pandas_generator = df[cols].itertuples(index = False,name = None )
                    cur.executemany(insert_sql,pandas_generator)
        except Exception as e:
            print(e)

            
        return


    def pull_options_data_from_database_per_expiration(self, ticker, target_date, expiration_date, conn_params):
        print("here")
        #extract options data from database
        sql_query = '''SELECT ticker, strike, midpoint, expiration, price_date, option_type
                       FROM options WHERE ticker = %s AND expiration = %s AND price_date = %s
                       ORDER BY strike ASC'''
        
        args = [ticker, expiration_date, target_date]
        
        try:
            with psycopg2.connect(**conn_params) as conn:
                with conn.cursor() as cur:
                    df = pd.read_sql(sql_query, conn, params = args)
        except Exception as e:
            print(e)


        #pulling risk free rate from database and broadcasting to pandas dataframe
        sql_query = '''SELECT risk_free_rate FROM market_data WHERE date = %s'''
        args = [target_date]

        try:
            with psycopg2.connect(**conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql_query,args)
                    results = cur.fetchall()
        except Exception as e:
            print(e)
            
        df["risk_free"] = results[0][0]


        #pulling EOD stock data and broadcasting to pandas dataframe
        sql_query = '''SELECT close FROM stock_data WHERE date = %s AND ticker = %s'''
        args = [target_date, ticker]
        try:
            with psycopg2.connect(**conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql_query,args)
                    results = cur.fetchall()
        except Exception as e:
            print(e)


        df["stock_price"] = results[0][0]

        #converting expiration date to date fraction and broadcasting to pandas dataframe
        date_fraction = self.date_calculator.dates_to_expiration_fraction(expiration_date)
        df["date_fraction"] = date_fraction

        #going to replace with real dividend yield, but need to calculate in database to handle it. 
        df['dividend_yield'] = 1/100

        return df
    
    def pull_data_and_calc_black_scholes_imp_vol(self, ticker, target_date, expiration_date, conn_params ):
        options_data = self.pull_options_data_from_database_per_expiration( ticker, target_date, expiration_date, conn_params)
        
        self.black_scholes_calculation(options_data, conn_params)

        return
    

    #calculates and stores implied vol
    def black_scholes_calculation(self, options_dataframe, conn_params):
        def call_or_put(arg_string):
            return self.black_scholes.call_or_put_method[arg_string]
        
        options_dataframe['risk_free'] = options_dataframe['risk_free']/100

        options_dataframe['call_or_put_func'] = options_dataframe['option_type'].map(call_or_put)

        options_dataframe['implied_vol'] = np.vectorize(self.black_scholes.newton_raphson_method_black_scholes)\
                                        (1e-5, options_dataframe['stock_price'],\
                                        options_dataframe['midpoint'],\
                                        options_dataframe['risk_free'],\
                                        options_dataframe['dividend_yield'],\
                                        options_dataframe['date_fraction'],\
                                        options_dataframe['strike'],\
                                        options_dataframe['call_or_put_func'])
        


        #Check values to see if it looks good. 
        '''
        call_df = options_dataframe[options_dataframe['option_type'] == "CALL"]
        put_df = options_dataframe[options_dataframe['option_type'] == 'PUT']

        print("printing df call rows")
        for row in call_df[['strike','implied_vol']].itertuples():
            print(row)

        print("printing df put rows")
        for row in put_df[['strike','implied_vol']].itertuples():
            print(row)'''
        

        #clean IV values to remove non-convergent numbers and replace with 0.
        options_dataframe['implied_vol'] = options_dataframe['implied_vol'].mask((options_dataframe['implied_vol']<0) | \
                                                                  (options_dataframe['implied_vol']>4), 0)
        
        
        #Store in database
        sql_query = '''INSERT INTO options (ticker, expiration, price_date, strike, option_type, bs_implied_vol) 
                       VALUES (%s, %s, %s, %s, %s, %s)
                       ON CONFLICT (ticker, expiration, price_date, strike, option_type)
                       DO UPDATE SET
                       bs_implied_vol = EXCLUDED.bs_implied_vol'''
        
        columns = ['ticker', 'expiration', 'price_date', 'strike', 'option_type', 'implied_vol']

        pd_generator = options_dataframe[columns].itertuples(index = False, name = None)


        try:
            with psycopg2.connect(**conn_params) as conn:
                with conn.cursor() as cur:
                    cur.executemany(sql_query, pd_generator)
        except Exception as e:
            print(e)


        '''
        #check to see if inserted properly
        sql_query = 'SELECT bs_implied_vol FROM options'

        try:
            with psycopg2.connect(**conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql_query)
                    results = cur.fetchall()
        except Exception as e:
            print(e)

        print("testing database rows")
        for row in results:
            print(row)
        
        return'''
        del options_dataframe
        return



    

    def pull_expiration_list_from_database(self,ticker,conn_params):

        retrieve_dates = '''SELECT dates from expiration_series WHERE ticker = %s; '''

        #pull date list from postgres database
        try:
            with psycopg2.connect(**conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute(retrieve_dates, [ticker])
                    dates_tuple = cur.fetchall()[0]
        except Exception as e:
            print(e)

        return dates_tuple
    

    #scrapes data from API for a list of expirations, ignores invalid expiration dates and stores data in database
    def scrape_data_interest_rate_data(self):
        return
    
    
    #Given a target date, pulls expiration data through the available options chain expirations
    #pulling data from API and storing in the database

    def iterate_through_expirations_load_data(self, ticker, target_date, conn_params):
        #pull the list of expiration dates from the database

        expiration_list = self.pull_expiration_list_from_database(ticker,conn_params)[0]
        # filter dates beyond target date
        # dates are filtered with all dates > cutoff included. Since this is EOD data, I am not including
        # the target day. We only want options contracts expiring afterwards.

        expiration_list = [date for date in expiration_list if date > target_date.date()]

        #for date in expiration_list, check if it's in the database
        #if it's not in the database, query the API
        #if it's not in the API, continue to the next date

        active_expirations = []
        previous_expirations = []

        for date in expiration_list:

            exists_query = '''SELECT (EXISTS ( SELECT 1 
                              FROM options
                              WHERE expiration = %s
                              AND price_date = %s
                              AND bs_implied_vol IS NOT NULL
                              ))::int;'''

            args = [date, target_date]
            try:
                with psycopg2.connect(**conn_params) as conn:
                    with conn.cursor() as cur:
                        cur.execute(exists_query, args)
                        results = cur.fetchall()[0][0]
            except Exception as e:
                print(f"DB Error: {e}")

            #explicit do nothing on 1, we already have data no need to pull more
            if results == 1:
                previous_expirations.append(date)
                print(date, 1)
            #If result = 0, we need to pull data from the API
            if results == 0:
                try:
                    self.options_api_pull_per_exp_date(ticker, target_date, date, conn_params)
                    active_expirations.append(date)
                    print(date, 0)
                except:
                    print(date, "not in API")
        return [previous_expirations, active_expirations]
    
    def calc_options_surface_for_date(self, ticker, target_date, conn_params):

        previous_dates, active_dates = self.iterate_through_expirations_load_data( ticker, target_date, conn_params)

        active_dates = [dt.datetime.strftime(date,"%Y-%m-%d") for date in active_dates]
        for date in active_dates:
            self.pull_data_and_calc_black_scholes_imp_vol(ticker, target_date, date, conn_params)

        return
    
    #Pulls options expiration list from database
    #iterates through all expiration dates
    #Either pulls from the API or reads from the database, if the data is already there
    #Takes this data, formats it into the correct format
    #feeds it to the different volatility surface models
    def recreate_options_surface_from_database(self):

        return

    def plot_options_surface_from_database(self, ticker, target_date, low_strike_coef, high_str_coef, interp_method,\
                                            option_type, conn_params):

        #load data from database

        sql_query = '''SELECT expiration, strike, bid, ask, volume, bs_implied_vol 
                       FROM options WHERE price_date = %s 
                       AND ticker = %s
                       AND option_type = %s'''
        
        
        args = [target_date, ticker, option_type]

        stock_price_query = '''SELECT close FROM stock_data WHERE date = %s AND ticker = %s'''

        stock_args = [target_date, ticker]


        try:
            with psycopg2.connect(**conn_params) as conn:
                df = pd.read_sql_query(sql=sql_query,con=conn,params=args)
                with conn.cursor() as cur:
                    cur.execute(stock_price_query, stock_args)
                    stock_price = cur.fetchone()[0]
                    print(stock_price)
        except Exception as e:
            print(e)

        last_stock_price = stock_price

        lower_strike = last_stock_price*low_strike_coef
        high_strike = last_stock_price*high_str_coef

        log_moneyness_arr = []
        maturities = []
        implied_vols_arr = []


        # split into subgroup df's per expiration date
        grouped_df = df.groupby('expiration')

        for group_name, group_df in grouped_df:
            string_date = dt.datetime.strftime(group_name, "%Y-%m-%d")
            days_to_exp = self.date_calculator.num_dates_to_expir(string_date)

            df_mask = (~group_df['bs_implied_vol'].isna()) & \
                        (group_df['bs_implied_vol'] > 0) & \
                        (group_df['bs_implied_vol'] <= 2.0) & \
                        (group_df['strike'] >= lower_strike) & \
                        (group_df['strike'] <= high_strike) & \
                        (group_df['bid'] > 0) & (df['ask'] > 0) & \
                        (group_df['volume'] > 0)
            
            strikes = group_df.loc[df_mask,'strike'].values
            maturity = days_to_exp
            implied_vols = group_df.loc[df_mask, 'bs_implied_vol'].values

            if len(strikes) == 0:
                continue

            log_moneyness = np.log(strikes/last_stock_price)
            log_moneyness_arr.extend(log_moneyness)
            maturities.extend([maturity]*len(strikes))
            implied_vols_arr.extend(implied_vols)

        log_moneyness_arr = np.array(log_moneyness_arr)
        maturities = np.array(maturities)
        implied_vols_arr = np.array(implied_vols_arr)

        if len(log_moneyness_arr) == 0:
            print("no dp avaiable for plotting")
            return
        
        logm_grid = np.linspace(log_moneyness_arr.min(),log_moneyness_arr.max(),50)
        maturity_grid = np.linspace(maturities.min(),maturities.max(), 50)

        M_grid, LM_grid = np.meshgrid(maturity_grid, logm_grid)

        IV_grid = griddata(
            points = (maturities, log_moneyness_arr),
            values = implied_vols_arr,
            xi = (M_grid, LM_grid),
            method = 'linear'
        )

        fig = go.Figure(data = [go.Surface(
            x = M_grid,
            y = LM_grid,
            z = IV_grid,
            colorscale= 'Viridis',
            colorbar = dict(title = "Implied Volatility")
        )])
        
        fig.update_layout(
            title = f"Implied Vol Surface (Log-Moneyness) for {ticker}",
            scene=dict(
            xaxis_title='Days to Expiration',
            yaxis_title='Log-Moneyness ln(K/S)',
            zaxis_title='Implied Volatility',
        ),
        autosize=True,
        width=800,
        height=700)

        fig.show()


        return


    #Reads Pandas 
    def pull_options_data_from_database(self, ticker, target_date ):
        return

    #need to update this to store in database
    def get_expiration_list_options_ticker(self, ticker, conn_params):
        BASE_URL = "http://127.0.0.1:25503/v3"
        params = {'symbol': ticker}

        url = BASE_URL + '/option/list/expirations'

        data_to_store = []
        with httpx.stream("GET", url, params = params, timeout=60) as response:
            response.raise_for_status()
            iter_lines = response.iter_lines()
            #skip header
            next(iter_lines)
            for line in iter_lines:
                for row in csv.reader(io.StringIO(line)):
                    date = dt.datetime.strptime(row[1], '%Y-%m-%d').date()
                    data_to_store.append(date)
                
        insert_sql = '''INSERT INTO expiration_series (
        ticker , dates)
                    VALUES (%s, %s)
                    ON CONFLICT DO NOTHING;
        '''

        try:
            with psycopg2.connect(**conn_params) as conn:
                with conn.cursor() as cur:
                    arguments = (ticker, data_to_store)
                    cur.execute(insert_sql,arguments)
        except Exception as e:
            print(e)

        return



"""
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
    
    return """
    


def main():
    conn_params = {
        "host": "localhost",
        "database": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "port": "5432"
    }
    #load_dotenv()
    #api_key0 = os.getenv("API_KEY")
    #client = RESTClient(api_key = api_key0)
    #contracts = client.list_options_contracts(underlying_ticker="AAPL", limit=100)

    

    target_date = dt.datetime.strptime('2026-02-12', '%Y-%m-%d')
    expiration_date = "2026-12-18"
    thetadata_test = thetadata_options_scrape_EOD('AAPL', target_date)


    thetadata_test.get_expiration_list_options_ticker('AAPL',conn_params)
    #thetadata_test.options_api_pull_per_exp_date('AAPL',target_date,expiration_date,conn_params)
    stock_range_start_date = target_date
    stock_range_end_date = target_date + timedelta(days = 10)
    '''thetadata_test.pull_options_data_from_database_per_expiration('AAPL',target_date,expiration_date,\
                                                                                conn_params)'''
    
    #thetadata_test.pull_data_and_calc_black_scholes_imp_vol("AAPL", target_date, expiration_date, conn_params)
    #thetadata_test.iterate_through_expirations_load_data("AAPL",target_date,conn_params)
    thetadata_test.calc_options_surface_for_date('AAPL',target_date,conn_params)
    thetadata_test.plot_options_surface_from_database('AAPL', target_date, 0.2, 1.8,'cubic','CALL', conn_params)


    '''
    call_bin_options = y_finance_options_chain("BBWI", "BinTree Continuous Deriv", 100,30, "call")
    call_put_options = y_finance_options_chain("BBWI","BinTree Continuous Deriv", 100,30, "put" )
    call_options_scholes = y_finance_options_chain("BBWI", "Black Scholes",None,None,"call")
    put_options_scholes = y_finance_options_chain("BBWI", "Black Scholes", None, None, "put")

    call_bin_options.plot_imp_vol_surface()
    call_put_options.plot_imp_vol_surface()
    call_options_scholes.plot_imp_vol_surface()
    put_options_scholes.plot_imp_vol_surface()'''

if __name__ == "__main__":
    main()