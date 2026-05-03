from dotenv import load_dotenv
import psycopg2
import os
import pandas as pd
import openpyxl
from io import StringIO
import yfinance as yf
from datetime import date, timedelta
import datetime as dt
import time
import psycopg2.extras
import numpy as np
from typing import Dict
import holidays
import requests
from psycopg2 import sql
import httpx
import csv
from bs4 import BeautifulSoup
import io
import random
import json
import simfin as sf
from simfin.names import *


load_dotenv()
sf.set_api_key(os.getenv('SIM_FIN_KEY'))
sf.set_data_dir(r'C:\Users\steph\Desktop\Coding\SimFin')

#iterates S and P 500 and scrapes dividends
def iterate_simfin(start_date:dt.datetime.date = dt.datetime.today().date(), end_date:dt.datetime.date = dt.datetime.today().date()):
    df = sf.load(dataset = 'shareprices', variant = 'daily', market = 'us', refresh_days = 1)

    df['Date'] = pd.to_datetime(df['Date'])

    df_div = scrape_simfin_dividends(df,start_date,end_date, conn_params)
    print(df_div.columns)
    shares_df = scrape_simfin_outstanding_shares(df, start_date, end_date, conn)
    print(shares_df.columns)

    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            store_dividends_in_database(cur, df_div, start_date, end_date, conn_params)
            store_outstanding_shares_in_database(cur, shares_df,)

    return

#Scrape dividend yields per date range. Gives Dividend in terms of dollars
def scrape_simfin_dividends(df: pd.DataFrame, ticker:str, start_date:dt.datetime.date = dt.datetime.today().date(),\
                             end_date:dt.datetime.date = dt.datetime.today().date()) -> pd.DataFrame:

    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date) & (df['Ticker'] == ticker) & (df['Dividend'].notna())]
    df = df[['Date','Dividend']]

    return df

def scrape_simfin_outstanding_shares(df: pd.DataFrame, ticker: str, start_date: dt.datetime.date = dt.datetime.today().date(),\
                                      end_date:dt.datetime.date = dt.datetime.today().date()) -> pd.DataFrame:
    
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date) & (df['Ticker'] == ticker)]
    df = df[['Date','Close']]

    return df

def store_dividends_in_database(cur, df:pd.DataFrame,ticker:str, start_date: dt.datetime.date = dt.datetime.today().date(),\
                                end_date:dt.datetime.date = dt.datetime.today().date(), conn_params = None):
    
    sql_insert = '''INSERT INTO stock_data (date, ticker, dividend )'''
    
    pandas_generator = df.itertuples(index=False, name=None)

    try:
        for row in pandas_generator:
            cur.execute()
    except Exception as e:
        print(e)
                


def store_outstanding_shares_in_database(cur, df:pd.DataFrame, start_date: dt.datetime.date = dt.datetime.today().date(),\
                                end_date:dt.datetime.date = dt.datetime.today().date(), conn_params = None):
    pandas_generator = df.itertuples(index=False, name = None)
    try:
        for row in pandas_generator:
            cur.execute()

    except Exception as e:
        print(e)

if __name__ == "__main__":
    conn_params = {
    "host": "localhost",
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": "5432"
    }
    print(sf.__version__)
    income = sf.load(dataset='income', variant='quarterly', market='us')
    balance = sf.load(dataset='balance', variant='quarterly', market='us')
    cashflow = sf.load(dataset='cashflow', variant = 'quarterly', market = 'us')
    share_prices = sf.load(dataset = 'shareprices', variant = 'daily', market = 'us')
    
    '''
    print("printing dividends")
    print("columns",share_prices.columns)
    print(share_prices[(share_prices['Ticker']=='AAPL')&(share_prices['Dividend'].notna())][['Date','Close','Dividend']].tail())

    aapl_div = cashflow[cashflow['Ticker'] == 'AAPL'][['Report Date','Publish Date','Restated Date', 'Dividends Paid','Shares (Basic)']].sort_values('Report Date')
    aapl_div = aapl_div.assign(Dividends_Paid_USD = aapl_div['Dividends Paid']/1_000_000)
    print(aapl_div[['Report Date','Publish Date','Restated Date', 'Dividends Paid','Shares (Basic)']].tail(12))'''