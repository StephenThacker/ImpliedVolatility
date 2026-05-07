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
from collections import defaultdict
import json
import simfin as sf
from simfin.names import *



#Contains "one-time load" scripts related to initalizing database tables or transferring CSV/API historical data to databases
#i.e., pull interest rates from CSV file and store in database

load_dotenv()
sf.set_api_key(os.getenv('SIM_FIN_KEY'))
sf.set_data_dir(r'C:\Users\steph\Desktop\Coding\SimFin')

def add_outstanding_shares_column(conn_params):
    alter_query = '''ALTER TABLE stock_data
                     ADD COLUMN IF NOT EXISTS shares_outstanding BIGINT'''
    
    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(alter_query)
                conn.commit()
    except Exception as e:
        print(e)

    return


def read_s_and_p_tickers_from_CSV(conn_params):

    df = pd.read_csv(
        r"C:\Users\steph\Desktop\Coding\Github Quant Finance\CSV_FILES\SP.csv")
    
    df = df['Symbol']
    tickers = df.tolist()

    alter_query = '''ALTER TABLE market_data
                     ADD COLUMN IF NOT EXISTS S_and_P_tickers TEXT[] '''


    today  = dt.datetime.today().date()
    store_csv_query = '''INSERT INTO market_data (date, S_and_P_tickers) VALUES (%s, %s)
                         ON CONFLICT (date) DO UPDATE SET
                         S_and_P_tickers = EXCLUDED.S_and_P_tickers'''
    
    args = [today, tickers]

    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(alter_query)
                cur.execute(store_csv_query, args)
                conn.commit()

                
    except Exception as e:
        print(e) 

    return


def add_div_percentage_to_table(conn_params):
    SQL_QUERY = ''' ALTER TABLE stock_data 
                    ADD COLUMN div_yield_per DOUBLE PRECISION DEFAULT 0 NOT NULL;'''
    
    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(SQL_QUERY)
                conn.commit()

                
    except Exception as e:
        print(e)   

    return

def alter_stock_data(conn_params):

    SQL_command = """ALTER TABLE stock_data ADD COLUMN special_dividend DOUBLE PRECISION DEFAULT 0;"""

    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(SQL_command)
                conn.commit()
        
    except Exception as e:
        print(e)




    

def add_imp_vol_columns_to_table(conn_params):

    SQL_QUERY = ''' ALTER TABLE options 
                    ADD COLUMN bs_implied_vol DOUBLE PRECISION,
                    ADD COLUMN bin_imp_vol DOUBLE PRECISION   '''
    

    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(SQL_QUERY)
                conn.commit()

                
    except Exception as e:
        print(e)



def initalize_expirations_table(conn_params):

    SQL_QUERY = '''CREATE TABLE IF NOT EXISTS expiration_series (
                   ticker VARCHAR(10),
                   dates  DATE[] NOT NULL,
                   PRIMARY KEY (ticker)
     
    
    )'''


    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(SQL_QUERY)
                conn.commit()

                
    except Exception as e:
        print(f"DB Error: {e}")
        
    return


#EOD Options data
def initalize_options_table(conn_params):


    SQL_QUERY = """ CREATE TABLE IF NOT EXISTS options (
                    ticker VARCHAR(10),
                    symbol VARCHAR(50),
                    expiration DATE,
                    strike DOUBLE PRECISION,
                    option_type VARCHAR(5),
                    created TIMESTAMP,
                    price_date DATE,
                    last_trade TIMESTAMPTZ,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume BIGINT,
                    count BIGINT,
                    bid_size BIGINT,
                    bid_exchange BIGINT,
                    bid DOUBLE PRECISION,
                    bid_condition BIGINT,
                    ask_size BIGINT,
                    ask_exchange BIGINT,
                    ask DOUBLE PRECISION,
                    ask_condition BIGINT,
                    midpoint DOUBLE PRECISION,
                    bs_implied_vol DOUBLE PRECISION DEFAULT 0,
                    bin_imp_vol DOUBLE PRECISION DEFAULT 0,


                    PRIMARY KEY (ticker, expiration, price_date, strike, option_type)
                    )

                    """

    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(SQL_QUERY)
                conn.commit()

                
    except Exception as e:
        print(f"DB Error: {e}")



def initialize_stock_data_table(conn_params):
    create_table = '''CREATE TABLE IF NOT EXISTS stock_data (
       ticker VARCHAR(10),
       date DATE,
       dividend DOUBLE PRECISION DEFAULT 0,
       div_yield_per DOUBLE PRECISION DEFAULT 0,
       close DOUBLE PRECISION DEFAULT 0,
       open DOUBLE PRECISION DEFAULT 0,
       high DOUBLE PRECISION DEFAULT 0,
       low DOUBLE PRECISION DEFAULT 0,
       volume BIGINT DEFAULT 0,
       special_dividend DOUBLE PRECISION DEFAULT 0,
       outstanding_shares BIGINT DEFAULT 0,


       PRIMARY KEY (ticker, date)
       )
       '''
    
    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(create_table)
                conn.commit()

                
    except Exception as e:
        print(f"DB Error: {e}")

    return

def get_S_and_P_master(conn_params):

    return


#S&P Master is a master list of the S and P, it's a starting point
# We scrape changes to the S & P and these are used to determine the S&P and any time before or after the Master

def initialize_general_market_data_table(conn_params):
    create_table = '''CREATE TABLE IF NOT EXISTS market_data (
       date DATE,
       risk_free_rate DOUBLE PRECISION,
       S_and_P_tickers TEXT[],
       S_and_P__master TEXT[] DEFAULT NONE,
       S_and_P_changes TEXT[] DEFAULT NONE,
       
       PRIMARY KEY (date)
       )
       '''
    
    
    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(create_table)
                conn.commit()

                
    except Exception as e:
        print(f"DB Error: {e}")

    return


def store_interest_rates_in_db(conn_params):
    df = pd.read_excel(
        r'C:\Users\steph\Desktop\Coding\Github Quant Finance\CSV_FILES\Interest_Rates_DB.xlsx',
        sheet_name='Results')

    insert_sql = '''INSERT INTO market_data  (date , risk_free_rate)\
                    VALUES ( %s, %s)
                    ON CONFLICT DO NOTHING;
    '''

    df["Effective Date"] = pd.to_datetime(df["Effective Date"]).dt.date

    cols = ['Effective Date', 'Rate (%)']

    df = df[cols].where(df[cols].notnull(),None)

    pandas_generator = df[cols].itertuples(index=False, name=None)

    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                pandas_generator = df[cols].itertuples(index = False,name = None )
                cur.executemany(insert_sql,pandas_generator)
            conn.commit()
    except Exception as e:
        print(e)            

    sql_query = """SELECT * FROM market_data"""  

    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(sql_query)
                results = cur.fetchall()
    except Exception as e:
        print(e)

    for row in results:
        print(row)

    return

#stores stock dividends from date range into database
def store_stock_dividends_yfinance(ticker, date_start,date_end, conn_params):
    try:
        stock = yf.Ticker(ticker)
        dividends = stock.dividends
    except Exception as e:
        print(f"Error fetching dividends for {ticker}: {e}")
        return 
    
    if dividends.empty:
        print("no dividends at all in stock")
        return 
    
    dividends = dividends.copy()
    if dividends.index.tz is not None:
        dividends.index = dividends.index.tz_localize(None)
    try:
        start_date = pd.to_datetime(date_start)
        end_date = pd.to_datetime(date_end)
        dividends = dividends.loc[start_date:end_date]
        if dividends.empty:
            print("no dividends within specific date range")
            return 
        
    
    except Exception as e:
        print(f"Date filtering error for {ticker} ({date_start} to {date_end}): {e}")
        return 
    
    if isinstance(dividends,pd.DataFrame):
        div_values = dividends['Dividends'].values
    else:
        div_values = dividends.values

    data = {
        'ticker': ticker,
        'date': dividends.index.date,
        'dividend': div_values,
    }

    df = pd.DataFrame(data)
    print("heree heree")

    insert_sql = """
        INSERT INTO stock_data (ticker, date, dividend)
        VALUES (%s, %s, %s)
        ON CONFLICT (ticker, date)
        DO UPDATE SET dividend = EXCLUDED.dividend;
    """

    rows_affected = 0
    pandas_generator = df.itertuples(index = False, name = None)
    print(next(pandas_generator))
    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                cur.executemany(insert_sql,pandas_generator)
                rows_affected = cur.rowcount
            conn.commit()
    except Exception as e:
        print(f"Database error while storing dividends for {ticker}: {e}")

    print(f"Successfully stored {rows_affected} dividend record(s) for {ticker}")
    return rows_affected



    #dividends = stock.dividends.loc[date_start,date_end]"""

    #stores EOD stock price history for last 10 years for a single stock ticker
def store_stock_price_history_yfinance(ticker,start_date = None, end_date = None, conn_params=None):

    stock = yf.Ticker(ticker)
    try:
        stock_price = stock.history(period="10y")
    except Exception as e:
        print(e)
        return
    
    if stock_price.empty:
        print("no data found")
        return
   


    df_data = {"ticker": ticker, "date": stock_price.index.date, "close": stock_price["Close"], "open": stock_price["Open"], \
                "high": stock_price["High"],"low": stock_price["Low"], "volume": stock_price["Volume"]}
    
    df = pd.DataFrame(df_data)

    if start_date is not None and end_date is not None:
        start_date = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
        end_date = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    df = df.where(pd.notnull(df),None)




    insert_sql = '''INSERT INTO stock_data 
                    (ticker, date, close, open, high, low, volume)
                    VALUES %s
                    ON CONFLICT (ticker, date)
                    DO UPDATE SET
                    close = EXCLUDED.close,
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    volume = EXCLUDED.volume'''
    
    pandas_generator = [tuple(x) for x in df.to_numpy()]


    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(cur,insert_sql,pandas_generator,page_size = 1000)
            conn.commit()
    except Exception as e:
        print(e)


    #Check if worked
    
    query_first = "SELECT * FROM stock_data WHERE ticker = %s ORDER BY date ASC LIMIT 100;"
    query_last = "SELECT * FROM stock_data WHERE ticker = %s ORDER BY date DESC LIMIT 100;"

    args_first = [ticker]
    args_second = [ticker]

    try:
            with psycopg2.connect(**conn_params) as conn:
                df_first = pd.read_sql_query(query_first, conn, params=args_first)
                
                df_last = pd.read_sql_query(query_last, conn, params= args_second)

                print(df_first)
                
                print(df_last.sort_values('date'))

    except Exception as e:
        print(e)
    

def calculate_and_store_dividend_yields_database(ticker, start_date, end_date, conn_params):

    #check database to make sure there is stock data
    exists_query= '''SELECT EXISTS (
                     SELECT 1 
                     FROM stock_data 
                     WHERE ticker = %s 
                     AND close IS NOT NULL
                     );'''
    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(exists_query, (ticker,))
                results =  cur.fetchone()[0]
    except Exception as e:
        print("No data for ticker")
        return
    
    if results == 0:
        print("no stock data for ticker")
        return
    
    if type(start_date) == 'str':
        start_date_dt = dt.datetime.strptime(start_date,"%Y-%m-%d").date() - timedelta(days=400)

    if type(end_date) == 'str':
        end_date_dt = dt.datetime.strptime(end_date, "%Y-%m-%d").date()

    if type(start_date) == dt.datetime or type(end_date)==dt.datetime:
        start_date_dt = start_date_dt - timedelta(days = 400)
        start_date_dt = start_date_dt.date()
        end_date_dt = end_date.date()

    

    #pull stock data query

    stock_data_query = '''SELECT date, close, dividend FROM stock_data WHERE ticker = %s
                          AND date >= %s AND date <= %s AND
                          close IS NOT NULL ORDER BY date ASC'''

    args = [ticker, start_date_dt, end_date_dt]

    try:
        with psycopg2.connect(**conn_params) as conn:
            stock_df = pd.read_sql_query(stock_data_query, conn, params = args)

    except Exception as e:
        print(e)
        return

    stock_df['date'] = pd.to_datetime(stock_df['date'])

    stock_df['dividend'] = stock_df['dividend'].fillna(0)

    stock_df['div_sum'] = stock_df.rolling(window = '365D', on = 'date')['dividend'].sum()

    stock_df['div_yield_per'] = (stock_df['div_sum']/stock_df['close'])
    
    stock_df['ticker'] = ticker

    stock_df = stock_df[stock_df['div_yield_per']>0]

    update_data = [(row.div_yield_per, row.ticker, row.date.date()) for row in stock_df.itertuples(index = False)]

    update_query = '''UPDATE stock_data
                      SET div_yield_per = %s
                      WHERE ticker = %s
                      AND date = %s'''
    
    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                cur.executemany(update_query, update_data)
            conn.commit()
        print(f"Successfully updated {len(update_data)} yield records for {ticker}.")
    except Exception as e:
        print(f"Database update failed for {ticker}: {e}")

    
    query_first = "SELECT date, div_yield_per FROM stock_data WHERE ticker = %s ORDER BY date ASC LIMIT 10;"
    query_last = "SELECT date, div_yield_per FROM stock_data WHERE ticker = %s ORDER BY date DESC LIMIT 10;"

    args_first =   [ticker]
    args_second = [ticker]

    try:
            with psycopg2.connect(**conn_params) as conn:
                df_first = pd.read_sql_query(query_first, conn, params=args_first)
                
                df_last = pd.read_sql_query(query_last, conn, params= args_second)

                print(df_first)
                
                print(df_last.sort_values('date'))

    except Exception as e:
        print(e)

    return

def iterate_through_S_and_P_store_dividend_yields(start_date, end_date,conn_params):

    #load S&P tickers from database
    ticker_query = '''SELECT S_and_P_tickers FROM market_data WHERE S_and_P_tickers IS NOT NULL
                      ORDER BY date DESC LIMIT 1'''

    try:
        with psycopg2.connect(**conn_params) as conn:
            df = pd.read_sql_query(ticker_query, conn)

    except Exception as e:
        print(e)
    
    tickers = df.iloc[-1].tolist()[0]

    for ticker in tickers:
        calculate_and_store_dividend_yields_database(ticker,start_date, end_date,conn_params=conn_params)
        print(ticker)
        #trying not to get rate limited by API
    return


def iterate_through_S_and_P_store_stock_values( start_date = None, end_date = None,conn_params = None):

    #load S&P tickers from database
    ticker_query = '''SELECT S_and_P_tickers FROM market_data WHERE S_and_P_tickers IS NOT NULL
                      ORDER BY date DESC LIMIT 1'''

    try:
        with psycopg2.connect(**conn_params) as conn:
            df = pd.read_sql_query(ticker_query, conn)

    except Exception as e:
        print(e)
    
    tickers = df.iloc[-1].tolist()[0]

    for ticker in tickers:
        store_stock_price_history_yfinance(ticker,start_date = start_date, end_date = end_date,conn_params=conn_params)
        print(ticker)
        #trying not to get rate limited by API
        time.sleep(0.5)
    return

def iterate_through_S_and_P_store_dividends(start_date, end_date, conn_params):

  
    #load S&P tickers from database
    ticker_query = '''SELECT S_and_P_tickers FROM market_data WHERE S_and_P_tickers IS NOT NULL 
                      ORDER By date DESC 
                      LIMIT 1'''

    try:
        with psycopg2.connect(**conn_params) as conn:
            df = pd.read_sql_query(ticker_query, conn)

    except Exception as e:
        print(e)
    
    tickers = df.iloc[-1].tolist()[0]
    print(tickers)

    for ticker in tickers:
        store_stock_dividends_yfinance(ticker, start_date, end_date, conn_params=conn_params)
        print(ticker)
        #trying not to get rate limited by API
        time.sleep(0.5)
    return


def S_and_P_tickers(conn_params):

    ticker_query = '''SELECT S_and_P_tickers FROM market_data WHERE S_and_P_tickers IS NOT NULL 
                      ORDER By date DESC 
                      LIMIT 1'''

    try:
        with psycopg2.connect(**conn_params) as conn:
            df = pd.read_sql_query(ticker_query, conn)

    except Exception as e:
        print(e)
    
    tickers = df.iloc[-1].tolist()[0]
    return tickers

def nightly_store_stock_price_for_S_and_P(conn_params, start_date = None, end_date = None):
    if start_date is None and end_date is None:
        today = dt.datetime.today()
        start_date = dt.datetime.strftime(today.date(), "%Y-%m-%d")
        end_date = start_date

    iterate_through_S_and_P_store_stock_values(start_date,end_date, conn_params)

    return

def store_nightly_interest_rate(conn_params):

    url = (
        "https://markets.newyorkfed.org/read"
        "?productCode=50"
        "&eventCodes=520"
        "&limit=30"
        "&startPosition=0"
        "&sort=postDt:-1"
        "&format=json"
    )
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if not data.get("refRates"):
            raise ValueError("No SOFR data returned")

        sql_insert = '''INSERT INTO market_data (date, risk_free_rate) VALUES (%s, %s)
                         ON CONFLICT (date) DO UPDATE SET
                         risk_free_rate = EXCLUDED.risk_free_rate'''
        print("1")
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                for row in data['refRates']:
                    rate = row['percentRate']
                    effective_date = dt.datetime.strptime(row["effectiveDate"], "%Y-%m-%d").date()
                    args = [effective_date, rate]
                    cur.execute(sql_insert, args)
            conn.commit()
    except Exception as e:
        print(e) 
        return
    
    return
def get_expiration_list_options_ticker(ticker, conn_params, base_url = "http://127.0.0.1:25503/v3" ):
    BASE_URL = base_url
    params = {'symbol': ticker}
    print("3")
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

def scrape_finviz_dividend_json(ticker):
    url = f"https://finviz.com/quote.ashx?t={ticker}&ta=1&p=d&ty=dv"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() 
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        script_tag = soup.find('script', id='route-init-data')
        
        if not script_tag:
            print("Could not find the script tag with id 'route-init-data'.")
            return None, None
            
        json_text = script_tag.string
        data = json.loads(json_text)
        
        dividends_data = data.get('dividendsData', [])
        dividends_annual = data.get('dividendsAnnualData', [])
        
        df_dividends = pd.DataFrame(dividends_data)
        df_annual = pd.DataFrame(dividends_annual)
        
        if not df_dividends.empty and 'Exdate' in df_dividends.columns:
            df_dividends['Exdate'] = pd.to_datetime(df_dividends['Exdate'])
            
        return df_dividends, df_annual

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Finviz: {e}")
        return None, None
    except json.JSONDecodeError as e:
        print(f"Error parsing the JSON data: {e}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None
    
def store_finviz_dividend(ticker, conn_params):

    df_1, df_2 = scrape_finviz_dividend_json(ticker)

    if df_1 is None or df_1.empty:
        print("no dividend data avaiable for ",ticker)
        return
    
    names_dict = {'Ticker': 'ticker', 'Exdate': 'date', 'Ordinary':'dividend', 'Special':'special_dividend'}

    df_1 = df_1.rename(columns=names_dict)

    iterator = df_1.itertuples(index= False, name = None)


    sql_query = '''INSERT INTO stock_data (ticker,date, dividend, special_dividend) VALUES (%s, %s, %s, %s)
                   ON CONFLICT (ticker, date) DO UPDATE SET dividend = EXCLUDED.dividend, 
                   special_dividend = EXCLUDED.special_dividend'''

    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                for row in iterator:
                    print('row', row)
                    cur.execute(sql_query, row)
    except Exception as e:
        print(e)


def scrape_finviz_for_dividend_data(conn_params):

    tickers = S_and_P_tickers(conn_params)
    for ticker in tickers:
        time.sleep(random.uniform(4,12))
        try:
            store_finviz_dividend(ticker, conn_params)
        except Exception as e:
            print(e)
        print(ticker)

def get_sp500_constituents_master(
    url: str = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies") -> list[str]:

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/129.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Find the table immediately after "S&P 500 component stocks" heading
    heading = soup.find(
        "span",
        class_="mw-headline",
        string=lambda text: text and "S&P 500 component stocks" in text,
    )

    if heading:
        table = heading.find_next("table")
    else:
        # Robust fallback using the official table ID
        table = soup.find("table", {"id": "constituents"})

    if not table:
        raise ValueError("Could not find the S&P 500 constituents table on the page.")

    tickers: list[str] = []
    # Skip the header row (first <tr>)
    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if not cells:
            continue

        # The ticker is in the first <td>, inside an <a class="external text"> link
        ticker_cell = cells[0]
        ticker_link = ticker_cell.find("a", class_="external text")

        if ticker_link:
            ticker = ticker_link.get_text(strip=True)
        else:
            # Fallback in case the link structure changes
            ticker = ticker_cell.get_text(strip=True)

        if ticker and ticker != "Symbol":   # avoid any leftover header text
            tickers.append(ticker)

    return tickers

def store_S_and_P_master(conn_params):
    tickers = get_sp500_constituents_master()
    insert_SQL = '''INSERT INTO market_data (date, s_and_p__master) VALUES (%s, %s)
                    ON CONFLICT (date) DO UPDATE 
                    SET s_and_p__master = EXCLUDED.s_and_p__master;'''

    date = dt.datetime.today().date()

    args = (date, tickers)

    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            cur.execute(insert_SQL, args)
        conn.commit()


    SQL_Query = '''SELECT s_and_p__master FROM market_data WHERE date = %s'''
    args = [date]

    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            cur.execute(SQL_Query, args)
            results = cur.fetchall()

    for row in results[0]:
        print(row)

    return



def get_sp500_changes():
   
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
    }
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    headline = soup.find("span", id=lambda x: x and "Selected_changes_to_the_list_of_S" in x)
    if not headline:
        raise ValueError("Could not find the 'Selected changes' section on Wikipedia.")
    
    h2 = headline.find_parent("h2")
    if not h2:
        raise ValueError("Could not find the heading for S&P 500 changes.")
    
    table = h2.find_next_sibling("table", class_="wikitable")
    if not table:
        for t in soup.find_all("table", class_="wikitable"):
            text = t.get_text().lower()
            if "date" in text and "added" in text and "removed" in text:
                table = t
                break
    
    if not table:
        raise ValueError("Could not find the S&P 500 changes table on Wikipedia.")
    
    changes = defaultdict(lambda: {"+": [], "-": []})
    
    rows = table.find_all("tr")
    
    for row in rows[2:]:  # skip the two header rows
        cells = row.find_all("td")
        if len(cells) < 4:
            continue
        
        date_str = cells[0].get_text(strip=True)
        if not date_str or date_str.lower() in ("", "date", "nan"):
            continue
        
        try:
            date_obj = dt.datetime.strptime(date_str, "%B %d, %Y").date()
        except ValueError:
            continue
        
        added_cell = cells[1].get_text(strip=True)
        if added_cell and added_cell.lower() not in ("nan", "—", "–", ""):
            ticker = added_cell.split(maxsplit=1)[0].strip().upper()
            if ticker:
                changes[date_obj]["+"].append(ticker)
        
        removed_cell = cells[3].get_text(strip=True)
        if removed_cell and removed_cell.lower() not in ("nan", "—", "–", ""):
            ticker = removed_cell.split(maxsplit=1)[0].strip().upper()
            if ticker:
                changes[date_obj]["-"].append(ticker)
    
    result = {}
    for date_obj, data in changes.items():
        signed_tickers = [f"+{t}" for t in data["+"]] + [f"-{t}" for t in data["-"]]
        if signed_tickers:
            result[date_obj] = ",".join(signed_tickers)   # <-- no spaces here
    
    return result


def store_S_and_P_changes(conn_params):

    return




def load_expiration_dates_all_tickers(conn_params, base_url = "http://127.0.0.1:25503/v3"):
    tickers = S_and_P_tickers(conn_params)
    for ticker in tickers:
        print("ticker", ticker)
        try:
            get_expiration_list_options_ticker(ticker, conn_params,base_url= "http://host.docker.internal:25503/v3")
            time.sleep(0.1)
        except Exception as e:
            print(e)

    return




def nightly_routine(conn_params):
    #adding logic for testing
    potential_day = dt.datetime.today()
    potential_day = potential_day.date()
    if potential_day.weekday() >= 5:
        print("not a market day")
        return 
        
    nyse_holidays = holidays.financial_holidays('NYSE')

    if potential_day in nyse_holidays:
        print("not a market day")
        return
    
    potential_day = dt.datetime.strftime(potential_day, "%Y-%m-%d")
    #load_expiration_dates_all_tickers(conn_params,"http://host.docker.internal:25503/v3")
    store_nightly_interest_rate(conn_params)
    nightly_store_stock_price_for_S_and_P(conn_params)
    iterate_through_S_and_P_store_dividends(potential_day,potential_day,conn_params)
    iterate_through_S_and_P_store_dividend_yields(potential_day,potential_day,conn_params)
    return




if __name__ == "__main__":
    conn_params = {
    "host": "localhost",
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": "5432"
    }

    
    #add_outstanding_shares_column(conn_params)

    results = get_sp500_changes()
    for d,b in results.items():
        print(d,b)
    #df1, df2 = scrape_finviz_dividend_json("BKE")

    #scrape_finviz_for_dividend_data(conn_params)
    #print("finviz routine finished")

    #iterate_through_S_and_P_store_dividend_yields()


    '''
    start_date = '2018-01-01'
    soft_start_date = dt.datetime.strptime(start_date,"%Y-%m-%d") - timedelta(days = 700)
    soft_start_date = dt.datetime.strftime(soft_start_date.date(), "%Y-%m-%d")
    end_date_dt = dt.datetime.today().date()
    end_date = dt.datetime.strftime(end_date_dt, "%Y-%m-%d")
    iterate_through_S_and_P_store_dividends(start_date, end_date, conn_params)
    iterate_through_S_and_P_store_dividend_yields(start_date, end_date,conn_params)'''
