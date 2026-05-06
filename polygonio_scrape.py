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
import csv
from bs4 import BeautifulSoup
from collections import defaultdict
import io
import random
import json
from utils import S_and_P_tickers

load_dotenv()

def scrape_short_interest_data():
    return


def scrape_dividend_data(ticker: str, start_date: dt.datetime = None, end_date: dt.datetime = None)->Dict:

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
        print("Pull failed")
        print(response.status_code,": ", response.text)


def store_divs_in_database_polyio(cur,ticker, results: Dict):

    results = results['results']

    dividend_dict = defaultdict(float)

    for row in results:
        key = (row['ex_dividend_date'], row['ticker'])
        amount = float(row['cash_amount'])
        dividend_dict[key] += amount

    dividend_list = [(date_val,tkr,amount) for (date_val,tkr), amount in dividend_dict.items()]

        
    sql_insert = '''INSERT INTO stock_data (date, ticker, dividend ) VALUES %s
                    ON CONFLICT (date,ticker) DO UPDATE SET
                    dividend = EXCLUDED.dividend'''
    

    if dividend_list:
        try:
            psycopg2.extras.execute_values(cur, sql_insert, dividend_list, page_size=2000)
        except Exception as e:
            print("Error storing dividends:", e)
    else:
        print(ticker, " No dividends found")


    #manaul_test_query = '''SELECT date,ticker, dividend FROM stock_data WHERE ticker = %s AND dividend > 0 ORDER by date DESC LIMIT 10'''




def pull_div_data_poly_for_all(conn_params, start_date: dt.datetime = None, end_date:dt.datetime = None):

    tickers = S_and_P_tickers(conn_params)
    if not start_date:
        start_date = dt.datetime.today()
    if not end_date:
        end_date = dt.datetime.today()

    successful_tickers  = []
    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                for ticker in tickers:
                    print(ticker)
                    time.sleep(0.1)
                    try:
                        results = scrape_dividend_data(ticker, start_date, end_date)
                    except Exception as e:
                        print(e)
                        print("trying again")
                        time.sleep(30)
                        try:
                            results = scrape_dividend_data(ticker, start_date, end_date)
                            print(ticker, "retry completed successfully")
                        except:
                            time.sleep(120)
                            print("trying one last time")
                            try:
                                results = scrape_dividend_data(ticker, start_date, end_date)
                            except:
                                print("couldn't get it to work, continuing")
                                continue
                            print(ticker, "retry completed successfully")
                    store_divs_in_database_polyio(cur, ticker, results)
                    if results['results']:
                        successful_tickers.append(ticker)

    except Exception as e:
        print("falled to pull dividend: ", e)
    
    return tickers




if __name__ == "__main__":
    conn_params = {
    "host": "localhost",
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": "5432"
    }
    start_date = dt.datetime.today() - timedelta(days=5000)
    end_date = dt.datetime.today() - timedelta(days = 1)
    pull_div_data_poly_for_all(conn_params, start_date, end_date)