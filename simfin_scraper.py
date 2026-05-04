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
from utils import S_and_P_tickers


load_dotenv()
sf.set_api_key(os.getenv('SIM_FIN_KEY'))
sf.set_data_dir(r'C:\Users\steph\Desktop\Coding\SimFin')

#iterates S and P 500 and scrapes dividends
def iterate_simfin(start_date:dt.datetime.date = dt.datetime.today().date(), end_date:dt.datetime.date = dt.datetime.today().date(), conn_params = None):
    df = sf.load(dataset = 'shareprices', variant = 'daily', market = 'us', refresh_days = 1)

    df['Date'] = pd.to_datetime(df['Date'])
    available_tickers = set(df['Ticker'].unique())
    #Need to update S_and_P_tickers function at a later date
    tickers = S_and_P_tickers(conn_params)
    print(tickers)
    print("finished reading tickers")
    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            for ticker in tickers:
                if ticker not in available_tickers:
                    print(ticker, " not found in SimFin Dataframe, skipping")
                    continue
                else:
                    print(ticker)
                df_div = scrape_simfin_dividends(df, ticker, start_date,end_date)
                shares_df = scrape_simfin_outstanding_shares(df, ticker, start_date, end_date)
                store_dividends_in_database(cur, ticker, df_div)
                store_outstanding_shares_in_database(cur,ticker, shares_df)

    #Checking if data is in database:
    #sql_query = '''SELECT (ticker, dividend, shares_outstanding) FROM stock_data WHERE date >= %s AND date <= %s '''
    '''
    results = None
    args = [start_date, end_date]
    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(sql_query,args)
                results = cur.fetchall()
    except Exception as e:
        print(e)

    for row in results:
        print(row)
    return'''

#Scrape dividend yields per date range. Gives Dividend in terms of dollars
def scrape_simfin_dividends(df: pd.DataFrame, ticker:str, start_date:dt.datetime.date = dt.datetime.today().date(),\
                             end_date:dt.datetime.date = dt.datetime.today().date()) -> pd.DataFrame:

    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date) & (df['Ticker'] == ticker) & (df['Dividend'].notna())]
    df = df[['Date','Ticker', 'Dividend']]

    return df

def scrape_simfin_outstanding_shares(df: pd.DataFrame, ticker: str, start_date: dt.datetime.date = dt.datetime.today().date(),\
                                      end_date:dt.datetime.date = dt.datetime.today().date()) -> pd.DataFrame:
    
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date) & (df['Ticker'] == ticker)]
    df = df[['Date','Ticker', 'Shares Outstanding' ]]

    return df

def store_dividends_in_database(cur,ticker, df:pd.DataFrame):
        
    sql_insert = '''INSERT INTO stock_data (date, ticker, dividend ) VALUES %s
                    ON CONFLICT (date,ticker) DO UPDATE SET
                    dividend = EXCLUDED.dividend'''
    
    dividend_list = list(df.itertuples(index=False, name=None))

    if dividend_list:
        try:
            psycopg2.extras.execute_values(cur, sql_insert, dividend_list, page_size=2000)
            #print(ticker, ": stored dividends")
        except Exception as e:
            print("Error storing dividends:", e)
    else:
        print(ticker, " No dividends found")
                


def store_outstanding_shares_in_database(cur,ticker, df:pd.DataFrame):
    
    sql_insert = '''INSERT INTO stock_data (date, ticker, shares_outstanding) VALUES %s
                    ON CONFLICT (date, ticker) DO UPDATE SET
                    shares_outstanding = EXCLUDED.shares_outstanding'''

    shares_outstanding_list = list(df.itertuples(index=False, name = None))

    if shares_outstanding_list:
        try:
            psycopg2.extras.execute_values(cur,sql_insert, shares_outstanding_list)
            #print(ticker, ": stored shares outstanding")
        except Exception as e:
            print(e)
    else:
        print(ticker, ": No datafound for shares_outstanding in dataframe ")


if __name__ == "__main__":
    conn_params = {
    "host": "localhost",
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": "5432"
    }
    #income = sf.load(dataset='income', variant='quarterly', market='us',refresh_days = 1)
    #balance = sf.load(dataset='balance', variant='quarterly', market='us',refresh_days = 1)
    #cashflow = sf.load(dataset='cashflow', variant = 'quarterly', market = 'us',refresh_days = 1)
    share_prices = sf.load(dataset = 'shareprices', variant = 'daily', market = 'us',refresh_days = 1)
    print(share_prices.columns)
    start_date =  dt.datetime.today()-timedelta(days=365)
    end_date = dt.datetime.today() - timedelta(days=150)
    #iterate_simfin(start_date = start_date, end_date = end_date)

    iterate_simfin(start_date, end_date, conn_params)