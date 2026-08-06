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
from numba import types
from numba.typed import Dict
import httpx
import io
from datetime import date, timedelta
import csv
import holidays
from utils import get_S_and_P_composite
import asyncio
from collections.abc import Iterator
import plotly
from tests import conftests
import implied_vol
from testcontainers.postgres import PostgresContainer
from data_helpers.ephemeral_db import start_test_db
from implied_vol import binomial_tree_vectorized, thetadata_options_scrape_EOD, calculate_dates



class binomial_tree_vellekoop():

    def __init__(self, number_of_layers, initial_stock_price, interest_rate,
                 time_to_expiration, stock_dividend, call_or_put,
                 target_date=None, conn_params = None, ticker = None, last_date = None, expiration_date = None):
        


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
        date_object = calculate_dates()
        self.targ_date = target_date
        self.days_to_expir = int(time_to_expiration)
        self.dividend_df = self.build_dividends_dataframe(conn_params, ticker, target_date, last_date)
        self.last_date = last_date
        self.dividend_tups_list = self.refine_dividends_list(target_date, expiration_date)
        self.indices = [elem[0] for elem in self.dividend_tups_list]
        self.divs = [elem[1] for elem in self.dividend_tups_list]
        self.div_dict = {int(k) : v for k,v in self.dividend_tups_list}


 
    def build_dividends_dataframe(self, conn_params,ticker:str, start_date:dt.datetime, end_date:dt.datetime) -> pd.DataFrame:
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
    
    
    def refine_dividends_list(self, start_date: dt.datetime, end_date:dt.datetime) -> list[tuple[int,float]]:
        if self.dividend_df is None or self.dividend_df.empty:
            return []
        
        filtered_divs = self.dividend_df.loc[(self.dividend_df['date'] >= start_date) & (self.dividend_df['date']<= end_date)]
        indices = filtered_divs['date'].map(self.convert_days_to_index).values
        divs = filtered_divs['dividend'].values
        filtered_tups = zip(indices, divs)

        return list(filtered_tups)
    
    def convert_days_to_index(self, dividend_ex_date):

        diff = (dividend_ex_date - self.targ_date).days

        total = self.days_to_expir


        if total == 0:
            return 0

        index = int(round((diff/total)*(self.number_of_layers-1)))

        return index
    
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
                       AND date_of_creation = (
                       SELECT MAX(date_of_creation)
                       FROM future_predictions
                       WHERE ticker = %s)
                       ORDER BY future_date ASC'''
        
        args = [ticker, start_date, end_date, ticker]
        
        try:
            with psycopg2.connect(**conn_params) as conn:
                df = pd.read_sql_query(sql_query, conn, params=args)
                return df
        except Exception as e:
            print(f"Error pulling future dividends: {e}")
            return pd.DataFrame()

        
    @staticmethod
    @njit(fastmath = True)
    def forward_pass_njit(number_of_layers, initial_stock_price, down_factor, up_factor):
        price_array = np.zeros((number_of_layers,number_of_layers))
        price_array[0,0] = initial_stock_price
        for i in range(1,number_of_layers):
            price_array[i,0] = price_array[i-1,0]*down_factor
            price_array[i,1:i+1] = price_array[i-1,0:i]*up_factor
        return price_array

    def backwards_pass_njit2(self, price_array,number_of_layers,discount_up,discount_down,strike, call_or_put, div_dict):
        options_array  = np.zeros((number_of_layers,number_of_layers))
        if call_or_put == True:
            options_array[-1,:] = np.maximum(price_array[-1,:] - strike, 0)
        if call_or_put == False:
            options_array[-1,:] = np.maximum(strike - price_array[-1,:], 0)
        for i in range(number_of_layers -2, -1,-1):

            continuation = discount_up*options_array[i+1,1:i+2] + discount_down*options_array[i+1,0:i+1]
            if i in div_dict.keys():
                if call_or_put == True:
                    expnd_continuation = [0.0] + list(continuation)
                else:
                    expnd_continuation = [strike] + list(continuation)
                ex_div_continuation = self._quotient_calc2(price_array,div_dict[i],i, expnd_continuation)
                continuation = ex_div_continuation

            intrinsic = np.maximum(price_array[i,0:i+1] - strike,0) if call_or_put == True else np.maximum(strike - price_array[i,0:i+1],0)
            options_array[i,0:i+1] = np.maximum(continuation,intrinsic)

        return options_array[0,0]
    


    def _quotient_calc2(self, price_array, dividend, index, continuation):

        stock_price_layer = price_array[index]
        stock_price_layer_zeros = np.array([float(stock_price_layer[i]) for i in range(len(stock_price_layer)) if stock_price_layer[i] != 0])
        div_subtract = np.maximum(stock_price_layer_zeros - float(dividend),0.0)

        stock_price_layer_full = [0] + stock_price_layer_zeros

        quotient_list = np.zeros(shape = len(stock_price_layer_zeros))

        for i in range(0,len(div_subtract)):
            for j in range(0,len(stock_price_layer_full)-1):
                if div_subtract[i] >= stock_price_layer_full[j] and div_subtract[i] <= stock_price_layer_full[j+1]:
                    try:
                        quotient_list[i] = continuation[j]+ (continuation[j+1] - continuation[j])*(div_subtract[i] - stock_price_layer_full[j])/(stock_price_layer_full[j+1] \
                                                                                          - stock_price_layer_full[j])
                    except Exception as e:
                        print(e)
                        quotient_list[i] = 0
                    break
                
        return quotient_list
    

    def backwards_pass_njit(self, price_array,number_of_layers,discount_up,discount_down,strike, call_or_put, div_dict):
        options_array  = np.zeros((number_of_layers,number_of_layers))
        if call_or_put == True:
            options_array[-1,:] = np.maximum(price_array[-1,:] - strike, 0)
        if call_or_put == False:
            options_array[-1,:] = np.maximum(strike - price_array[-1,:], 0)
        for i in range(number_of_layers -2, -1,-1):

            continuation = discount_up*options_array[i+1,1:i+2] + discount_down*options_array[i+1,0:i+1]
            if i in div_dict.keys():
                if call_or_put == True:
                    expnd_continuation = [0.0] + list(continuation)
                else:
                    expnd_continuation = [strike] + list(continuation)
                ex_div_continuation = self._quotient_calc(price_array,div_dict[i],i, expnd_continuation)
                continuation = ex_div_continuation

            intrinsic = np.maximum(price_array[i,0:i+1] - strike,0) if call_or_put == True else np.maximum(strike - price_array[i,0:i+1],0)
            options_array[i,0:i+1] = np.maximum(continuation,intrinsic)

        return options_array[0,0]
    


    def _quotient_calc(self, price_array, dividend, index, continuation):

        stock_price_layer = price_array[index]
        stock_price_layer_zeros = np.array([float(stock_price_layer[i]) for i in range(len(stock_price_layer)) if stock_price_layer[i] != 0])
        div_subtract = np.maximum(stock_price_layer_zeros - float(dividend),0.0)

        stock_price_layer_full = [0] + stock_price_layer_zeros

        quotient_list = np.zeros(shape = len(stock_price_layer_zeros))

        for i in range(0,len(div_subtract)):
            for j in range(0,len(stock_price_layer_full)-1):
                if div_subtract[i] >= stock_price_layer_full[j] and div_subtract[i] <= stock_price_layer_full[j+1]:
                    try:
                        quotient_list[i] = continuation[j]+ (continuation[j+1] - continuation[j])*(div_subtract[i] - stock_price_layer_full[j])/(stock_price_layer_full[j+1] \
                                                                                          - stock_price_layer_full[j])
                    except Exception as e:
                        print(e)
                        quotient_list[i] = 0
                    break
                
        return quotient_list


    
    def pricing_forward_pass(self,sigma, strike):
        numba_dict = Dict.empty(key_type = types.int64, value_type = types.float64)
        for key, value in self.div_dict.items():
            numba_dict[key] = value

        print(numba_dict)
        print(self.div_dict)

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

        
        return self.backwards_pass_njit(price_array,number_of_layers,discount_up,discount_down,strike, call_or_put,
                                        self.div_dict)


        
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
        


    #Can be done outside the optimization loop.
    # Vellekoop formula is C_{i,j} = V[i,m] + (V[i, m+1] - V[i,m]) * (S[i,j] - Dividend[i]) - S[i,m])/(S[i,m+1]- S[i,m])
    # where m is such that S[i,m] <= (S[i,j] - Div[i])<= S[i,m+1]
    # In other words, you subtract the dividend shift at the ex-dividend date n_{D}.
    # This creates an offset that breaks the recombining tree
    # So, given a specific node x in the n_{D} layer,
    # Just before the dividend is subtracted, we partition the stock prices by their nodes in the binomial trees at layer n_{D}, we denote
    # this partition as \{P_{i,j}\}, where P_{i,j} is the real valued interval \[S_{i,p}, S_{i,p+1}\]. We refer to P as the set of all partitions
    # We then subtract the dividend value from node value S_{i,j}.
    # At each node, we associate that node with the partition that the node lies in. 
    # Since it's possible that the dividend can be big enough to oversubtract, pushing x outside of all of the partitions,
    # We add an extra partition A =  \[0, S_{i,0}], where S_{i,0} is the lowest stock price in that layer of the binomial tree.
    # So the resulting structure is P Union A. 
    # We identify these with the j index  0, ,.,j,.., N
    # This function assigns the j index of the appropriate partitions to each node in the tree





    

def test_db_func(conn_params):

    select_SQL_statement = '''SELECT * FROM stock_data WHERE date = %s '''

    price_date = date(2026, 5, 18)

    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:

            cur.execute(select_SQL_statement, (price_date,))
            results = cur.fetchall()
    

    for row in results:
        print(row)


def plot_options_surface(ticker, strikes, implied_vols, days_to_exp, stock_price,
                         interp_method='linear',
                         fixed_logm_min=None, fixed_logm_max=None,
                         fixed_mat_min=None, fixed_mat_max=None):

    strikes = np.asarray(strikes, dtype=float)
    implied_vols = np.asarray(implied_vols, dtype=float)

    if np.isscalar(days_to_exp):
        maturities = np.full_like(strikes, float(days_to_exp))
    else:
        maturities = np.asarray(days_to_exp, dtype=float)

    last_stock_price = float(stock_price)

    # 1. Calculate log-moneyness ln(K / S) first so we can filter by it
    log_moneyness = np.log(strikes / last_stock_price)

    # 2. Apply new filters: IV <= 1.0, Expiration <= 350, Log-Moneyness between -1.0 and 0.5
    mask = (
        (~np.isnan(implied_vols)) & 
        (implied_vols > 0)
    )

    strikes = strikes[mask]
    implied_vols = implied_vols[mask]
    maturities = maturities[mask]
    log_moneyness = log_moneyness[mask]

    if len(strikes) == 0:
        print("No valid option data available for plotting after applying filters.")
        return

    # Create 2D interpolation grids
    logm_grid = np.linspace(
        fixed_logm_min if fixed_logm_min is not None else log_moneyness.min(),
        fixed_logm_max if fixed_logm_max is not None else log_moneyness.max(), 50)

    maturity_grid = np.linspace(
        fixed_mat_min if fixed_mat_min is not None else maturities.min(),
        fixed_mat_max if fixed_mat_max is not None else maturities.max(), 50)

    M_grid, LM_grid = np.meshgrid(maturity_grid, logm_grid)

    # Interpolate 3D IV grid across Days to Expiration and Log-Moneyness
    IV_grid = griddata(
        points=(maturities, log_moneyness),
        values=implied_vols,
        xi=(M_grid, LM_grid),
        method=interp_method
    )

    fig = go.Figure(data=[go.Surface(
        x=M_grid,
        y=LM_grid,
        z=IV_grid,
        colorscale='Viridis',
        colorbar=dict(title="Implied Volatility")
    )])

    fig.update_layout(
        title=f"Implied Volatility Surface (Log-Moneyness) for {ticker}",
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

    return [M_grid, LM_grid, IV_grid]

def write_datasample_to_file(data_sample, call_type):
    return

def plot_data_for_group(conn_params, ticker, target_date, expiration_list, call_or_put):

    all_strikes = []
    all_implied_vols = []
    all_days_to_exp = []
    latest_stock_price = None

    last_date = expiration_list[-1]

    for exp in expiration_list:
        current_data = get_data_per_expiration(conn_params,ticker, target_date, exp, call_or_put)

        stock_price = current_data['stock_price'].iloc[-1]
        latest_stock_price = stock_price
        interest_rate = current_data['risk_free'].iloc[-1]
        days_to_exp = current_data['days_to_expir'].iloc[-1]
        strikes = current_data['strike'].values
        midpoints = current_data['midpoint'].values

        IV_call_vals = generate_and_solve_tree_per_expiration(conn_params, 500, stock_price, interest_rate, days_to_exp, 
            ticker, last_date, exp, strikes, midpoints)

        all_strikes.extend(strikes)
        all_implied_vols.extend(IV_call_vals)
        all_days_to_exp.extend([days_to_exp] * len(strikes))

    if len(all_strikes) > 0 and latest_stock_price is not None:
        plot_options_surface(ticker, all_strikes,all_implied_vols, all_days_to_exp,latest_stock_price,interp_method='linear')

    return

def generate_and_solve_tree_per_expiration(conn_params,number_of_layers, stock_price,interest_rate,days_to_exp,ticker, last_date,exp_date ,strikes,  midpoints):
    call_tree = binomial_tree_vellekoop(number_of_layers=number_of_layers,
                        initial_stock_price=stock_price,
                        interest_rate=interest_rate,
                        time_to_expiration=days_to_exp,
                        stock_dividend=0,
                        call_or_put='PUT',
                        target_date=target_date,
                        conn_params=conn_params,
                        ticker=ticker,
                        last_date=last_date,
                        expiration_date=exp_date
                    )
    cal_vec_func = np.vectorize(call_tree.vectorized_brentq_wrapper, otypes=[float])
    IV_call_vals = cal_vec_func(0.01, 5.0, strikes, midpoints)


    return IV_call_vals

def get_data_per_expiration(conn_params, ticker, target_date,expiration, call_or_put:str):

    data_sample = theta_data_object.pulling_all_options_data_for_pricing(conn_params, ticker, target_date,expiration)

    if data_sample is None or data_sample.empty:
        raise ValueError("No data found")

    call_or_put_mask = data_sample['option_type'] == call_or_put

    filtered_data = data_sample[call_or_put_mask]

    
    return filtered_data


if __name__ == "__main__":
    postgres = None
    try:
        conn_params, postgres = start_test_db()

        ticker = 'CVX'
        target_date = date(2026, 5, 18)

        theta_data_object = thetadata_options_scrape_EOD()

        # Fetch available expiration dates
        expirations_list = theta_data_object.select_available_expiration_dates_for_ticker(conn_params, ticker, target_date)


        plot_data_for_group(conn_params, ticker, target_date, expirations_list, 'PUT')


    finally:
        if postgres is not None:
            postgres.stop()