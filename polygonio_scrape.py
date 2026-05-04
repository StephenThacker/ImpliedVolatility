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
import httpx
from psycopg2 import sql
import httpx
import csv
from bs4 import BeautifulSoup
import io
import random
import json
from utils import S_and_P_tickers

load_dotenv()

def scrape_short_interest_data():
    return

def scrape_dividend_data(ticker, start_date, end_date, conn_params):
    url = "https://api.massive.com/stocks/v1/dividends"


def scrape_dividend_data(ticker: str, start_date: dt.datetime = None, end_date: dt.datetime = None):

    if not start_date:
        start_date = dt.datetime.today()
    if not end_date:
        end_date = dt.datetime.today()

    start_date = dt.datetime.strftime(start_date, '%Y-%m-%d')
    end_date = dt.datetime.strftime(end_date, '%Y-%m-%d')


    url = "https://api.massive.com/stocks/v1/dividends"
    
    params = {
        "ticker": ticker,
        "limit": 100,
        "sort": "ticker.asc",
        "ex_dividend_date.gte": start_date,
        "ex_dividend_date.lte": end_date,
        "apiKey": os.getenv('POLYGONIO_KEY')}
    
    response = httpx.get(url,params = params, verify=False)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API request failed with status {response.status_code}: {response.text}")


def store_divs_in_database_polyio(cur,ticker, df:pd.DataFrame):
        
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


def pull_poly_io_data_per_ticker(ticker, conn_params):

    pass

def pull_div_data_poly_for_all(conn_params, start_date: dt.datetime, end_date:dt.datetime):
    tickers = S_and_P_tickers()
    

    pass



if __name__ == "__main__":
    start_date = dt.datetime.today() - timedelta(days=365)
    end_date = dt.datetime.today() - timedelta(days = 30)
    results = scrape_dividend_data('AAPL', start_date, end_date)
    for row in results['results']:
        print(row['pay_date'],row['cash_amount'])