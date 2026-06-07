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
from tests import conftests
import implied_vol
from testcontainers.postgres import PostgresContainer
from data_helpers.ephemeral_db import start_test_db
from implied_vol import binomial_tree_vectorized, thetadata_options_scrape_EOD, calculate_dates


#need try, finally code block here for database management of ephemeral database.


class binomial_tree_vellekoop(binomial_tree_vectorized):

    def __init__(self, number_of_layers, initial_stock_price, interest_rate,
                 time_to_expiration, stock_dividend, call_or_put,
                 target_date=None, conn_params = None, ticker = None, last_date = None):
        
        super().__init__(number_of_layers,initial_stock_price, interest_rate, time_to_expiration, stock_dividend, call_or_put)
        self.targ_date = target_date
        self.days_to_expir = int(time_to_expiration)
        self.dividend_df = self.build_dividends_dataframe(conn_params, ticker, target_date, last_date)
        dividend_tups_list = self.refine_dividends_list(target_date, last_date)
        print(dividend_tups_list)



 
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
    
    def refine_dividends_list(self, start_date: dt.datetime, end_date:dt.datetime) -> list[tuple[dt.date,float]]:
        if self.dividend_df is None or self.dividend_df.empty:
            return []
        
        filtered_divs = self.dividend_df.loc[(self.dividend_df['date'] >= start_date) & (self.dividend_df['date']<= end_date)]
        filtered_tups = filtered_divs.itertuples(index = False, name = None)
        return list(filtered_tups)
    

    
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


        
    
    def map_dividend_dates_to_integer_timestep(self,dividends_tup_list):
        return
    

    def forward_pass_python(self, number_of_layers, initial_stock_price, down_factor, up_factor):
        return super.forward_pass_njit(number_of_layers,initial_stock_price,down_factor,up_factor)
        
    
    def backwards_pass_python(self, price_array, number_of_layers, discount_up, discount_down, strike, call_or_put):
        return
    
    '''
    @staticmethod
    @njit(fastmath = True)
    def forward_pass_njit(number_of_layers, initial_stock_price, down_factor, up_factor):
        return 
    def forward_pass_njit(number_of_layers, initial_stock_price, down_factor, up_factor,dividend_tups):
        if not dividend_tups:
            #HAve to de-modularize code here, because using NJIT and does not support Python OOP
            price_array = np.zeros((number_of_layers,number_of_layers))
            price_array[0,0] = initial_stock_price
            for i in range(1,number_of_layers):
                price_array[i,0] = price_array[i-1,0]*down_factor
                price_array[i,1:i+1] = price_array[i-1,0:i]*up_factor
            return price_array
        else:
            pass

    @staticmethod
    @njit(fastmath = True)
    def backwards_pass_njit(price_array, number_of_layers, discount_up, discount_down, strike, call_or_put):
        return '''
    

    '''
    def backwards_pass_njit(price_array, number_of_layers, discount_up, discount_down, strike, call_or_put,dividend_tups):
        if not dividend_tups:
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
        else:
            pass'''
    

def test_db_func(conn_params):

    select_SQL_statement = '''SELECT * FROM stock_data WHERE date = %s '''

    price_date = date(2026, 5, 18)

    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:

            cur.execute(select_SQL_statement, (price_date,))
            results = cur.fetchall()
    

    for row in results:
        print(row)
        




if __name__ == "__main__":
    postgres = None
    try:
        conn_params, postgres = start_test_db()
        #print(conn_params)

        #test_db_func(conn_params)


        target_date = date(2026, 5, 18)


        theta_data_object = thetadata_options_scrape_EOD()

        expirations_list = theta_data_object.select_available_expiration_dates_for_ticker(conn_params, 'XOM', target_date)


        trial_date_no_div= expirations_list[0]
        print(type(trial_date_no_div))
        trial_date_divs = expirations_list[11]
        last_date = expirations_list[-1]

        data_sample_1 = theta_data_object.pulling_all_options_data_for_pricing(conn_params, 'XOM', target_date, trial_date_no_div)

        stock_price = data_sample_1['stock_price'].iloc[-1]
        interest_rate = data_sample_1['risk_free'].iloc[-1]
        stock_dividend_yield = 0 #Not relevant for this version
        days_to_expiration = data_sample_1['days_to_expir'].iloc[-1]
        call_tree = binomial_tree_vellekoop(100, stock_price,interest_rate,days_to_expiration, stock_dividend_yield, 'CALL', trial_date_no_div,\
                                            conn_params, 'XOM', last_date)




        data_sample_2 = theta_data_object.pulling_all_options_data_for_pricing(conn_params, 'XOM', target_date, trial_date_divs)

        stock_price_div = data_sample_2['stock_price'].iloc[-1]
        interest_rate_div = data_sample_2['risk_free'].iloc[-1]
        stock_dividend_yield = 0 #Not relevant for this version
        days_to_expiration_div = data_sample_2['days_to_expir'].iloc[-1]

        call_tree_divs = binomial_tree_vellekoop(100, stock_price_div, interest_rate_div, days_to_expiration_div,stock_dividend_yield,\
                                                 'CALL',target_date,conn_params, 'XOM', last_date)

        


    finally:
        if postgres is not None:
            postgres.stop()
