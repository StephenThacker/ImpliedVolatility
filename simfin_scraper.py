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

    scrape_simfin_dividends(df,start_date, end_date)

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

if __name__ == "__main__":
    pass
