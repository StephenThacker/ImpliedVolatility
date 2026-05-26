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
                 target_date=None):
        
        super().__init__(number_of_layers,initial_stock_price, interest_rate, time_to_expiration, stock_dividend, call_or_put)
        self.targ_date = target_date
        self.days_to_expir = int(time_to_expiration)


    def collect_dividends_per_tree(self)->list[tuple[dt.datetime,float]]:
        end_date = self.targ_date + timedelta(days=self.days_to_expir)
        dividend_df = self.dividend.loc[(self.dividend['date']>= self.targ_date)&(self.dividend['date']<= end_date)]
        if dividend_df.empty == True:
            return []
        else:
            iterable_df = dividend_df.itertuples(index = False, name = None)
            date_div_pairs = []
            for row in iterable_df:
                date_div_pairs.append(row)
            return date_div_pairs
        
    
    def map_dividend_dates_to_integer_timestep(self,dividends_tup_list):
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
        




if __name__ == "__main__":
    postgres = None
    try:
        conn_params, postgres = start_test_db()
        print(conn_params)


    finally:
        if postgres is not None:
            postgres.stop()
