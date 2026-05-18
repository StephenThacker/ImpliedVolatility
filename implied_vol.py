from dotenv import load_dotenv
import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf
from scipy.optimize import brentq
import plotly.graph_objects as go
import datetime as dt
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from psycopg2.extras import execute_values
import os
import psycopg2
import time
from numba import njit, prange
import httpx
import io
from datetime import date, timedelta
import csv
import holidays
from utils import get_S_and_P_composite
import asyncio
from collections.abc import Iterator
import plotly

load_dotenv()

class binomial_tree_vectorized:

    def __init__(self, number_of_layers, initial_stock_price, interest_rate, time_to_expiration, stock_dividend,call_or_put):
        self.number_of_layers = number_of_layers
        self.initial_stock_price = initial_stock_price
        self.time_to_expiration = time_to_expiration
        self.interest_rate = interest_rate
        self.dividend = stock_dividend
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
        call_or_put = self.call_or_put.lower()
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
        
    def vectorized_brentq_wrapper(self,sigma_low,sigma_high,strike_price,midpoint, xtol=1e-8, rtol=1e-8, maxiter=100):
        def brentq_objective(sigma):
            return self.vectorization_of_forward_pass(sigma,strike_price) - midpoint

        try:
            #start = time.perf_counter()
            result = brentq(brentq_objective, sigma_low, sigma_high, xtol=1e-8, rtol=1e-8, maxiter=100)
            #stop = time.perf_counter()
            #print("time brentq",stop-start)
            return result
        except ValueError:
            return np.nan
        

        
    # Recombining tree method that uses interpolation method to adjust for dividend distributions, while maintaining speed
    # "Extracting Information on Implied Volatilities and Discrete Dividends from American Option Prices" by Martina Nardon, Paolo Pianca
class binomial_tree_vellekoop(binomial_tree_vectorized):

    def __init__(self, *args, **kwargs):          # you can also list specific parameters
        super().__init__(*args, **kwargs)

    
    @staticmethod
    @njit(fastmath = True)
    def forward_pass_njit(number_of_layers, initial_stock_price, down_factor, up_factor):
        return 
    
    @staticmethod
    @njit(fastmath = True)
    def backwards_pass_njit(price_array, number_of_layers, discount_up, discount_down, strike, call_or_put):
        return 
    

    


    
class calculate_dates:

    def dates_to_expiration_fraction(self, expiration_date, target_date):
        expiration_date = dt.datetime.strptime(expiration_date, "%Y-%m-%d").date()
        target_date = dt.datetime.strptime(target_date, "%Y-%m-%d").date()
        days_to_expiration = (expiration_date - target_date).days
        fraction_of_days = days_to_expiration/365
        return fraction_of_days

    def dates_to_expiration_days(self,expiration_date, target_date):
        target_date = dt.datetime.strptime(target_date, "%Y-%m-%d").date()
        expiration_date = dt.datetime.strptime(expiration_date, "%Y-%m-%d").date()
        days_to_expiration = (expiration_date - target_date).days
        return days_to_expiration

    def num_dates_to_expir(self,expiration_date, target_date):
        target_date = dt.datetime.strptime(target_date, "%Y-%m-%d").date()
        expiration_date = dt.datetime.strptime(expiration_date, "%Y-%m-%d").date()
        days_to_expiration = (expiration_date - target_date).days
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


#target date is assummed to be a datetime object
class thetadata_options_scrape_EOD:
    def __init__(self):
        #might want to refactor calculate dates to be an instance passed to the class, to avoid redundant memory drag
        #If I'm going to pull data for like 500 tickers
        self.date_calculator = calculate_dates()
        self.black_scholes = black_scholes_implied_volatility()

   
    #selects expirations that currently exist in the database, for a specific target
    def select_available_expiration_dates_for_ticker(self, conn_params, ticker : str, target_date : dt.datetime.date)\
          -> list[dt.datetime.date]:
        sql_query = '''SELECT DISTINCT expiration FROM options WHERE ticker = %s AND price_date = %s ORDER BY expiration ASC'''
        args = [ticker, target_date]

        try:
            with psycopg2.connect(**conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql_query,args)
                    results = cur.fetchall()
        except Exception as e:
            print(e)

        date_list = []
        for row in results:
            date_list.append(row[0])
    
        return date_list
    
    
    #Please note that risk free rate is updated at 8AM in the morning, the morning after the target date
    #Therefore, for evaluating stocks during the same day, or night of, you have to use the previous day's rate.
    def pull_risk_free_rate_database(self, cur, target_date: dt.datetime.date) -> float:
        sql_query = sql_query = '''SELECT date, risk_free_rate 
                                   FROM market_data 
                                   WHERE date <= %s 
                                   ORDER BY date DESC 
                                   LIMIT 1'''

        args = [target_date]
        try:
            cur.execute(sql_query,args)
            row = cur.fetchone()
        except Exception as e:
            print(e)

        pulled_date = row[0]

        if pulled_date != target_date:
            print("WARNING: risk free rate does not match today's date")
        
        return row[1]/100
    
    def select_stock_data_for_pricing(self,cur, ticker:str, target_date: dt.datetime.date) -> float:
        sql_query = '''SELECT close, div_yield_per FROM stock_data WHERE ticker = %s AND date = %s'''
        args = [ticker, target_date]
        try:
            cur.execute(sql_query,args)
            results = cur.fetchone()
        except Exception as e:
            print(e)

        
        return [results[0],results[1]]
    
    def select_options_data_for_pricing(self, conn, ticker:str, target_date:dt.datetime.date, expiration_date: dt.datetime.date):
        sql_query = '''SELECT ticker, strike, midpoint, expiration, price_date, option_type FROM options WHERE
                       ticker = %s AND expiration = %s and price_date = %s ORDER by strike ASC'''
        
        args = [ticker, expiration_date, target_date]

        try:
            df = pd.read_sql(sql_query, conn, params = args)
        except Exception as e:
            print(e)

        
        return df
    
    def pulling_all_options_data_for_pricing(self, conn_params, ticker: str, target_date: dt.datetime.date, expiration_date: dt.datetime.date):

        try:
            with psycopg2.connect(**conn_params) as conn:
                df = self.select_options_data_for_pricing(conn,ticker, target_date, expiration_date)
                if df.empty:
                    print("No data found")
                    return
                with conn.cursor() as cur:
                    risk_free = self.pull_risk_free_rate_database(cur, target_date)
                    stock_close, div_yield = self.select_stock_data_for_pricing(cur, ticker, target_date)
        except Exception as e:
            print(e)

        target_date_str = dt.datetime.strftime(target_date, "%Y-%m-%d")
        exp_date_str = dt.datetime.strftime(expiration_date, "%Y-%m-%d")
        date_fraction = self.date_calculator.dates_to_expiration_fraction(exp_date_str, target_date_str)
        days_to_expiration = self.date_calculator.dates_to_expiration_days(exp_date_str, target_date_str)

        df['risk_free'] = risk_free
        df['dividend_yield'] = div_yield
        df['stock_price'] = stock_close
        df["date_fraction"] = date_fraction
        df['days_to_expir'] = days_to_expiration


        return df
    
    def build_options_surface_from_database_refactored(self, conn_params, ticker:str, target_date:dt.datetime.date, calculation_type:str, dividend_list = None):
        #selects expirations that currently exist in the database, for a specific target
        expirations_list = self.select_available_expiration_dates_for_ticker(conn_params, ticker, target_date)

        if calculation_type == 'Vellekoop':
            if not dividend_list:
                dividend_start_date = target_date
                dividend_end_date = expirations_list[-1]
                dividends_df = self.build_dividends_dataframe(conn_params, ticker, dividend_start_date, dividend_end_date)
                print("printing dividend ticker")
                print(dividends_df.head())
                print(dividends_df.tail())
            else:
                dividends_df = dividend_list            

        for expiration_date in expirations_list:
            options_dataframe = self.pulling_all_options_data_for_pricing(conn_params, ticker, target_date, expiration_date)
            
            if not isinstance(dividend_list, pd.DataFrame) or dividend_list.empty:
                options_dataframe = self.calculate_iv_surface_refactored(calculation_type, options_dataframe)
            else:
                options_dataframe = self.calculate_iv_surface_refactored(calculation_type, options_dataframe, dividends_df)

            options_dataframe = self.filter_iv_data(options_dataframe, -5, 15)
            self.store_iv_data(conn_params,options_dataframe, calculation_type)

        pass

    def build_options_surfaces_within_date_range(self, conn_params, ticker:str, start_date: dt.datetime, end_date:dt.datetime,\
                                                  calculation_type):
        
        if calculation_type == 'Vellekoop':
            dividend_start_date = start_date
            dividend_end_date = end_date + timedelta(days=1000)
            dividends_df = self.build_dividends_dataframe(conn_params, ticker, dividend_start_date, dividend_end_date)
        else:
            dividends_df = None



        current_date = start_date
        while current_date <= end_date:
            if not isinstance(dividends_df, pd.DataFrame) or dividends_df.empty:
                self.build_options_surface_from_database_refactored(conn_params, ticker,current_date.date(), calculation_type)
            else:
                self.build_options_surface_from_database_refactored(conn_params, ticker,current_date.date(), calculation_type, dividend_list = dividends_df)
            current_date = current_date + timedelta(days=1)

    def build_dividends_dataframe(self,conn_params,ticker:str, start_date:dt.datetime, end_date:dt.datetime):
        historical_df = self.pull_dividend_db(conn_params, ticker, start_date, end_date)
        future_df = self.pull_future_dividends_estimation(conn_params, ticker, start_date, end_date)

        dfs = []
        if historical_df is not None and not historical_df.empty:
            dfs.append(historical_df)
        if future_df is not None and not future_df.empty:
            dfs.append(future_df)
            
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df.sort_values('date', inplace=True)
            combined_df.reset_index(drop=True, inplace=True)
            return combined_df
        
        return pd.DataFrame(columns=['date', 'dividend'])
    
    def pull_dividend_db(self, conn_params,ticker:str, start_date:dt.datetime, end_date:dt.datetime):
        sql_query = '''SELECT date, dividend FROM stock_data 
                       WHERE ticker = %s AND date >= %s AND date <= %s AND dividend > 0
                       ORDER BY date ASC'''
        
        args = [ticker, start_date, end_date]
        
        try:
            with psycopg2.connect(**conn_params) as conn:
                df = pd.read_sql_query(sql_query, conn, params=args)
                return df
        except Exception as e:
            print(f"Error pulling historical dividends: {e}")
            return pd.DataFrame()

    def pull_future_dividends_estimation(self, conn_params,ticker:str, start_date:dt.datetime, end_date:dt.datetime):
        sql_query = '''SELECT future_date as date, estimated_dividend as dividend FROM future_predictions 
                       WHERE ticker = %s AND future_date >= %s AND future_date <= %s AND estimated_dividend > 0
                       ORDER BY future_date ASC'''
        
        args = [ticker, start_date, end_date]
        
        try:
            with psycopg2.connect(**conn_params) as conn:
                df = pd.read_sql_query(sql_query, conn, params=args)
                return df
        except Exception as e:
            print(f"Error pulling future dividends: {e}")
            return pd.DataFrame()

    def calculate_iv_surface_refactored(self, calculation_type:str, options_dataframe:pd.DataFrame,dividends_df=None, number_of_layers = 100, dividend_df = None) -> pd.DataFrame:
        def call_or_put(arg_string):
            
            return self.black_scholes.call_or_put_method[arg_string]
        
        options_dataframe['call_or_put_func'] = options_dataframe['option_type'].map(call_or_put)

        if calculation_type == 'Vellekoop':
            stock_price = options_dataframe['stock_price'].iloc[-1]
            interest_rate = options_dataframe['risk_free'].iloc[-1]
            dividends = dividends_df
            days_to_expiration = options_dataframe['days_to_expir'].iloc[-1]
            call_tree = binomial_tree_vellekoop(number_of_layers, stock_price, interest_rate, days_to_expiration, dividends, 'CALL')
            put_tree = binomial_tree_vellekoop(number_of_layers, stock_price, interest_rate, days_to_expiration,dividends,"PUT")
            
            cal_vec_func = np.vectorize(call_tree.vectorized_brentq_wrapper, otypes=[float])
            options_dataframe.loc[is_call, 'implied_vol'] = cal_vec_func(0.01, 5, options_dataframe.loc[is_call, 'strike'].values, options_dataframe.loc[is_call, 'midpoint'].values)     
            put_vec_func = np.vectorize(put_tree.vectorized_brentq_wrapper, otypes=[float])
            options_dataframe.loc[is_put, 'implied_vol'] = put_vec_func(0.01, 5, options_dataframe.loc[is_put, 'strike'].values, options_dataframe.loc[is_put, 'midpoint'].values)
            del call_tree
            del put_tree


        if calculation_type == "Black Scholes":
            options_dataframe['implied_vol'] = np.vectorize(self.black_scholes.newton_raphson_method_black_scholes)\
                                            (1e-5, options_dataframe['stock_price'],\
                                            options_dataframe['midpoint'],\
                                            options_dataframe['risk_free'],\
                                            options_dataframe['dividend_yield'],\
                                            options_dataframe['date_fraction'],\
                                            options_dataframe['strike'],\
                                            options_dataframe['call_or_put_func'])
        

        if calculation_type == "Binomial Tree":
            stock_price = options_dataframe['stock_price'].iloc[-1]
            interest_rate = options_dataframe['risk_free'].iloc[-1]
            stock_dividend_yield = options_dataframe['dividend_yield'].iloc[-1]
            days_to_expiration = options_dataframe['days_to_expir'].iloc[-1]
            call_tree = binomial_tree_vectorized(number_of_layers, stock_price, interest_rate, days_to_expiration, stock_dividend_yield,"CALL")
            put_tree = binomial_tree_vectorized(number_of_layers, stock_price, interest_rate, days_to_expiration, stock_dividend_yield,"PUT")

            #creating pandas masks
            is_call = options_dataframe['option_type'] == 'CALL'
            is_put = options_dataframe['option_type'] == 'PUT'
            
            
            cal_vec_func = np.vectorize(call_tree.vectorized_brentq_wrapper, otypes=[float])
            options_dataframe.loc[is_call, 'implied_vol'] = cal_vec_func(0.01, 5, options_dataframe.loc[is_call, 'strike'].values, options_dataframe.loc[is_call, 'midpoint'].values)     
            put_vec_func = np.vectorize(put_tree.vectorized_brentq_wrapper, otypes=[float])
            options_dataframe.loc[is_put, 'implied_vol'] = put_vec_func(0.01, 5, options_dataframe.loc[is_put, 'strike'].values, options_dataframe.loc[is_put, 'midpoint'].values)
            del call_tree
            del put_tree

        return options_dataframe
    
    def filter_iv_data(self,options_dataframe,lower_bound, upper_bound):
        options_dataframe['implied_vol'] = options_dataframe['implied_vol'].mask((options_dataframe['implied_vol']<lower_bound) | \
                                                                  (options_dataframe['implied_vol']>upper_bound), 0)
        return options_dataframe
    
    def store_iv_data(self,conn_params, options_dataframe, calculation_type):

        #Store in database
        if calculation_type == "Black Scholes":
            sql_query = '''INSERT INTO options (ticker, expiration, price_date, strike, option_type, bs_implied_vol) 
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (ticker, expiration, price_date, strike, option_type)
                        DO UPDATE SET
                        bs_implied_vol = EXCLUDED.bs_implied_vol'''
        
        if calculation_type == "Binomial Tree":
            sql_query = '''INSERT INTO options (ticker, expiration, price_date, strike, option_type, bin_imp_vol) 
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (ticker, expiration, price_date, strike, option_type)
                        DO UPDATE SET
                        bin_imp_vol = EXCLUDED.bin_imp_vol'''

            
        columns = ['ticker', 'expiration', 'price_date', 'strike', 'option_type', 'implied_vol']

        pd_generator = options_dataframe[columns].itertuples(index = False, name = None)


        try:
            with psycopg2.connect(**conn_params) as conn:
                with conn.cursor() as cur:
                    cur.executemany(sql_query, pd_generator)
        except Exception as e:
            print(e)


        del options_dataframe
        return


    
    def _format_options_ticker_for_API(self,ticker:str) -> str:
        #Removes decimals from strings, because ThetaData API does not have any tickers with decimals for options contracts
        return ticker.replace('.',"")
       
    

    #pull options data for a ticker between two dates
    def options_api_pull_refactored(self,ticker = None, start_date:dt.datetime = None, end_date: dt.datetime = None,\
                                    base_url = "http://127.0.0.1:25503/v3") -> Iterator[dict[str, str]]:
        
        
        start_date = dt.datetime.strftime(start_date.date(),"%Y%m%d")
        end_date = dt.datetime.strftime(end_date.date(),"%Y%m%d")
        ticker = self._format_options_ticker_for_API(ticker)

        BASE_URL = base_url

        PARAMS = {'start_date': start_date, 'end_date':end_date, 'symbol':ticker, "expiration":"*"}

        url = BASE_URL + '/option/history/eod'

        with httpx.stream("GET",url, params = PARAMS, timeout=60) as response:
            response.raise_for_status()
            lines = response.iter_lines()
            reader = csv.DictReader(lines)

            yield from reader

    

    #can potentially move db connection to outside the loop of S&P tickers, which would make it faster.
    #Probably worth doing, if I move from 500 tickers to a few thousand.
    def stream_options_into_db(self,ticker:str,  start_date: dt.datetime = dt.datetime.today(), end_date:dt.datetime = dt.datetime.today(), conn_params = None,
                               base_url:str = "http://127.0.0.1:25503/v3"):
        
        insert_sql = '''INSERT INTO options (
        ticker, expiration, strike, option_type, price_date,\
                open, high, low, close, volume, count, bid_size,bid,\
                    ask_size, ask, midpoint )
                    VALUES %s
                    ON CONFLICT (ticker, expiration, strike, price_date, option_type) DO NOTHING;
      
        '''
        batch_size = 1000
        batch_list = []
        try:
            with psycopg2.connect(**conn_params) as conn:
                with conn.cursor() as cur:
                    for row in self.options_api_pull_refactored(ticker,start_date, end_date, base_url):
                        values = (ticker,row['expiration'],row['strike'],row['right'],row['created'],\
                                   row['open'], row['high'], row['low'], row['close'], row['volume'], row['count'],\
                                     row['bid_size'], row['bid'],row['ask_size'], row['ask'], ((float(row['ask'])+float(row['bid']))/2) )
                        batch_list.append(values)
                        
                        if len(batch_list) >= batch_size:
                            execute_values(cur, insert_sql, batch_list)
                            batch_list = []
                    
                    if batch_list:
                        execute_values(cur,insert_sql,batch_list)

            return True

        except Exception as e:
            print(e)
            return False
    
    def stream_stock_data_into_db(self, ticker:str, start_date:dt.datetime, end_date: dt.datetime, conn_params,\
                                   base_url:str = "http://127.0.0.1:25503/v3"):
        insert_sql = '''INSERT into stock_data(
                        ticker, date, close, open, high, low, volume)
                        VALUES %s
                        ON CONFLICT (ticker, date) DO UPDATE SET
                            CLOSE = EXCLUDED.close,
                            OPEN = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            volume = EXCLUDED.volume;'''
        batch_size = 1000
        batch_list = []

        try:
            with psycopg2.connect(**conn_params) as conn:
                with conn.cursor() as cur:
                    for row in self.stream_stock_data_thetadata(ticker, start_date, end_date, base_url):
                        date = dt.datetime.strptime(row['created'].split('T')[0], "%Y-%m-%d")
                        values = (ticker, date, row['close'], row['open'],row['high'], row['low'], row['volume'])
                        batch_list.append(values)

                        if len(batch_list) >= batch_size:
                            execute_values(cur, insert_sql,batch_list)
                            batch_list = [] 

                    if batch_list:
                        execute_values(cur,insert_sql,batch_list)

                return True

        except Exception as e:
            print(e)
            return False
                        
                    

        return

    #Base url needs to be different for Docker
    def stream_stock_data_thetadata(self, ticker: str, start_date:dt.datetime, end_date:dt.datetime,\
                                     base_url:str = "http://127.0.0.1:25503/v3" )->Iterator[dict[str, str]]:
        
        
        start_date = dt.datetime.strftime(start_date.date(),"%Y%m%d")
        end_date = dt.datetime.strftime(end_date.date(),"%Y%m%d")

        ticker = ticker

        BASE_URL = base_url

        PARAMS = {'start_date': start_date, 'end_date':end_date, 'symbol':ticker, "expiration":"*"}

        url = BASE_URL + '/stock/history/eod'

        with httpx.stream("GET",url, params = PARAMS, timeout=60) as response:
            response.raise_for_status()
            lines = response.iter_lines()
            reader = csv.DictReader(lines)

            yield from reader  
    

    def plot_options_surface_from_database(self, ticker, target_date, low_strike_coef, high_str_coef, interp_method,
                                                option_type, conn_params, calculation_type, animate=False,
                                                fixed_logm_min=None, fixed_logm_max=None,
                                                fixed_mat_min=None, fixed_mat_max=None):

        if isinstance(target_date, dt.datetime):
            target_date = target_date.date()

        #load data from database
        if calculation_type == "Black Scholes":
            sql_query = '''SELECT expiration, strike, bid, ask, volume, bs_implied_vol AS implied_volatility
                        FROM options WHERE price_date = %s 
                        AND ticker = %s
                        AND option_type = %s'''
            
        if calculation_type == "Binomial Tree":
            sql_query = '''SELECT expiration, strike, bid, ask, volume, bin_imp_vol AS implied_volatility
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
            target_date_str = dt.datetime.strftime(target_date, "%Y-%m-%d")
            days_to_exp = self.date_calculator.num_dates_to_expir(string_date, target_date_str)

            
            df_mask = (~group_df['implied_volatility'].isna()) & \
                        (group_df['implied_volatility'] > 0) & \
                        (group_df['implied_volatility'] <= 5.0) 
            
            #df_mask = (~group_df['implied_volatility'].isna())
            
            strikes = group_df.loc[df_mask,'strike'].values
            maturity = days_to_exp
            implied_vols = group_df.loc[df_mask, 'implied_volatility'].values

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
        
        logm_grid = np.linspace(fixed_logm_min if fixed_logm_min is not None else log_moneyness_arr.min(),
                                fixed_logm_max if fixed_logm_max is not None else log_moneyness_arr.max(), 50)
        
        maturity_grid = np.linspace(fixed_mat_min if fixed_mat_min is not None else maturities.min(),
                                    fixed_mat_max if fixed_mat_max is not None else maturities.max(), 50)

        M_grid, LM_grid = np.meshgrid(maturity_grid, logm_grid)


        IV_grid = griddata(
            points = (maturities, log_moneyness_arr),
            values = implied_vols_arr,
            xi = (M_grid, LM_grid),
            method = interp_method
        )
        if animate == False:
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

        return [M_grid, LM_grid, IV_grid]


    def build_options_animation(self, ticker, start_date, end_date, low_strike_coef, high_str_coef, interp_method,
                                option_type, conn_params, calculation_type):

        
        nyse_holidays = holidays.financial_holidays('NYSE')
        date_list = []
        current = start_date
        while current <= end_date:
            if current.weekday() >= 5 or current in nyse_holidays:
                current += timedelta(days=1)
                continue
            date_list.append(current)
            current += timedelta(days=1)

        FIXED_LOGM_MIN = -1
        FIXED_LOGM_MAX = 1
        FIXED_MAT_MIN  = 0
        FIXED_MAT_MAX  = 365
        FIXED_Z_MIN    = 0.0
        FIXED_Z_MAX    = 1

        frames = []
        for date in date_list:
            try:
                M_grid, LM_grid, IV_grid = self.plot_options_surface_from_database(
                    ticker, date, low_strike_coef, high_str_coef, interp_method,
                    option_type, conn_params, calculation_type, animate=True,
                    fixed_logm_min=FIXED_LOGM_MIN,
                    fixed_logm_max=FIXED_LOGM_MAX,
                    fixed_mat_min=FIXED_MAT_MIN,
                    fixed_mat_max=FIXED_MAT_MAX
                )
            except Exception as e:
                print("date missing for imp vol surface", e)
                continue

            frame = go.Frame(
                data=[go.Surface(
                    x=M_grid,
                    y=LM_grid,
                    z=IV_grid,
                    colorscale='Viridis',
                    colorbar=dict(title='Implied Volatility'),
                    cmin=FIXED_Z_MIN,          # lock color scale
                    cmax=FIXED_Z_MAX
                )],
                name=date.strftime('%Y-%m-%d'),
                layout=dict(                   # ← lock axes in EVERY frame
                    scene=dict(
                        xaxis=dict(range=[FIXED_MAT_MIN, FIXED_MAT_MAX]),
                        yaxis=dict(range=[FIXED_LOGM_MIN, FIXED_LOGM_MAX]),
                        zaxis=dict(range=[FIXED_Z_MIN, FIXED_Z_MAX]),
                        camera=dict(
                            eye=dict(x=1.7, y=1.7, z=1.1),
                            up=dict(x=0, y=0, z=1),
                            center=dict(x=0, y=0, z=0)
                        )
                    )
                )
            )
            frames.append(frame)

        if not frames:
            print("No frames created")
            return None

        # Initial figure
        fig = go.Figure(
            data=[go.Surface(
                x=frames[0].data[0].x,
                y=frames[0].data[0].y,
                z=frames[0].data[0].z,
                colorscale='Viridis',
                colorbar=dict(title='Implied Volatility'),
                cmin=FIXED_Z_MIN,
                cmax=FIXED_Z_MAX
            )],
            frames=frames
        )

        fig.update_layout(
            title=f"{ticker} {option_type} Implied Volatility Surface Animation",
            scene=dict(
                xaxis=dict(range=[FIXED_MAT_MIN, FIXED_MAT_MAX],
                           title='Days to Expiration'),
                yaxis=dict(range=[FIXED_LOGM_MIN, FIXED_LOGM_MAX],
                           title='Log-Moneyness ln(K/S)'),
                zaxis=dict(range=[FIXED_Z_MIN, FIXED_Z_MAX],
                           title='Implied Volatility'),
                camera=dict(eye=dict(x=1.7, y=1.7, z=1.1))
            ),
            uirevision='keep_view',       
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Play", method="animate",
                         args=[None, {"frame": {"duration": 500, "redraw": True},
                                      "fromcurrent": True,
                                      "transition": {"duration": 200}}]),
                    dict(label="Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                        "transition": {"duration": 0}}])
                ]
            )],
            sliders=[dict(
                steps=[dict(method='animate',
                            args=[[frame.name], {"frame": {"duration": 300, "redraw": True},
                                                 "mode": "immediate"}],
                            label=frame.name) for frame in frames],
                transition={"duration": 300},
                x=0.1, xanchor="left", y=0.05, yanchor="bottom",
                currentvalue={"prefix": "Date: ", "visible": True}
            )]
        )
        return fig

    def scrape_stock_data_theta_data_S_and_P(self, start_date:dt.datetime, end_date:dt.datetime,conn_params: dict['str','str'],\
                                             base_url:str = "http://127.0.0.1:25503/v3"):
        tickers = get_S_and_P_composite(conn_params, start_date, end_date)
        failed_tickers = []
        for ticker in tickers:
            #Pause to avoid API rate limit
            time.sleep(0.06)
            print(ticker)
            success = self.stream_stock_data_into_db(ticker, start_date,end_date,conn_params, base_url)
            if not success:
                failed_tickers.append(ticker)
                print("added ", ticker," to failed_tickers")
        return
    
    
    def scrape_options_data_theta_data_S_and_P(self, start_date: dt.datetime, end_date: dt.datetime, conn_params: dict['str','str'],\
                                               base_url:str = "http://127.0.0.1:25503/v3"):
        tickers = get_S_and_P_composite(conn_params, start_date, end_date)
        failed_tickers = []
        for ticker in tickers:
            #Pause to avoid API rate limit
            time.sleep(0.06)
            print(ticker)
            success = self.stream_options_into_db(ticker, start_date, end_date,conn_params, base_url)
            #Theta data seems to parse options tickers differently than stock tickers in the situation where 
            #stock tickers have a "." in the ticker. For example, BRK.B is "BRK.B" for the stock API, but
            #"BRKB" for the options API. So, for any stock ticker that fails, we retry the request again, parsing out the "."
            if not success:
                if ticker == 'NVR':
                    print('NVR is an exception and has no options contracts')
                else:
                    failed_tickers.append(ticker)
                    print("added ", ticker," to failed_tickers")

        return
    
    def one_time_script_load_options_stock_data(self,start_date:dt.datetime, end_date:dt.datetime, conn_params:dict[str,str], base_url: str = "http://127.0.0.1:25503/v3" ):

        return
    
    def build_options_surface_entire_S_and_P(self, conn_params, start_date: dt.datetime, end_date: dt.datetime, calculation_type:str):
        tickers = get_S_and_P_composite(conn_params, start_date, end_date)

        for ticker in tickers:
            print(ticker)
            try:
                self.build_options_surfaces_within_date_range(conn_params, ticker, start_date, end_date,calculation_type)
            except Exception as e:
                print("couldn't build surface", e)

        return
    
def main():
    conn_params = {
        "host": "localhost",
        "database": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "port": "5432"
    }

    
    thetadata_test = thetadata_options_scrape_EOD()

    
    today = dt.datetime.today()
    end_date = dt.datetime.today() - timedelta(days=3)
    start_date = dt.datetime.today() - timedelta(days = 300)

    target_date_manual_test = dt.datetime.today() + timedelta(days=180)

    thetadata_test.build_options_surfaces_within_date_range(conn_params,'CVX',start_date,end_date, calculation_type= 'Vellekoop')


if __name__ == "__main__":
    main()
