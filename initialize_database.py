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
import io




#Contains "one-time load" scripts related to initalizing database tables or transferring CSV/API historical data to databases
#i.e., pull interest rates from CSV file and store in database

load_dotenv()


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

def one_time_dividend_update(conn_params):
    sql_update_query = '''UPDATE stock_data
       SET dividend = 0 WHERE dividend IS NULL'''
    
    sql_alter_query = '''ALTER TABLE stock_data
       ALTER COLUMN dividend SET DEFAULT 0'''
    
    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(sql_update_query)
                cur.execute(sql_alter_query)
               
                conn.commit()

                
    except Exception as e:
        print(e)   

    return
    

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
                    bs_implied_vol DOUBLE PRECISION,
                    bin_imp_vol DOUBLE PRECISION,


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
       dividend DOUBLE PRECISION,
       close DOUBLE PRECISION,
       open DOUBLE PRECISION,
       high DOUBLE PRECISION,
       low DOUBLE PRECISION,
       volume BIGINT,

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

def initialize_general_market_data_table(conn_params):
    create_table = '''CREATE TABLE IF NOT EXISTS market_data (
       date DATE,
       risk_free_rate DOUBLE PRECISION,
       
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
        r'C:\Users\steph\Desktop\Coding\Github Quant Finance\CSV_FILES\Interest_Rates_DB_old.xlsx',
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
    
    data = {
        'ticker': ticker,
        'date': dividends.index.date,
        'dividend': dividends.values,
    }

    df = pd.DataFrame(data)

    insert_sql = """
        INSERT INTO stock_data (ticker, date, dividend)
        VALUES (%s, %s, %s)
        ON CONFLICT (ticker, date)
        DO UPDATE SET dividend = EXCLUDED.dividend;
    """

    rows_affected = 0
    pandas_generator = df.itertuples(index = False, name = None)
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
    """
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
        print(e)"""
    

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

    start_date_dt = dt.datetime.strptime(start_date,"%Y-%m-%d").date() - timedelta(days=400)

    end_date_dt = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
    

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
        time.sleep(0.5)
    return


def iterate_through_S_and_P_store_stock_values(conn_params = None, start_date = None, end_date = None):

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
        store_stock_price_history_yfinance(ticker,start_date = None, end_date = None,conn_params=conn_params)
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

def nightly_store_stock_price_for_S_and_P(conn_params):
    todays_date = dt.datetime.strftime(dt.datetime.today().date(), "%Y-%m-%d")
    iterate_through_S_and_P_store_stock_values(conn_params,todays_date,todays_date)

    return

def store_nightly_interest_rate(conn_params):

    url = (
        "https://markets.newyorkfed.org/read"
        "?productCode=50"
        "&eventCodes=520"
        "&limit=1"
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

        latest = data["refRates"][0]
        effective_date = dt.datetime.strptime(latest["effectiveDate"], "%Y-%m-%d")
        rate = latest["percentRate"]

        sql_insert = '''INSERT INTO market_data (date, risk_free_rate) VALUES (%s, %s)
                         ON CONFLICT (date) DO UPDATE SET
                         risk_free_rate = EXCLUDED.risk_free_rate'''
        
        args = [effective_date, rate]
                        
    except Exception as e:
        print(e) 
        return

    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(sql_insert, args)
            conn.commit()
    except Exception as e:
        print(e)
        return

    return
def get_expiration_list_options_ticker(ticker, conn_params, base_url = "http://127.0.0.1:25503/v3" ):
    BASE_URL = base_url
    params = {'symbol': ticker}

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
    potential_day = dt.datetime.today().date()

    if potential_day.weekday() >= 5:
        print("not a market day")
        return 
        
    nyse_holidays = holidays.financial_holidays('NYSE')

    if potential_day in nyse_holidays:
        print("not a market day")
        return
    potential_day = dt.datetime.strftime(potential_day, "%Y-%m-%d")
    load_expiration_dates_all_tickers(conn_params,"http://host.docker.internal:25503/v3")
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

    #add_imp_vol_columns_to_table(conn_params)
    #store_interest_rates_in_db(conn_params)
    #initialize_stock_data_table(conn_params)
    #add_div_percentage_to_table(conn_params)
    #one_time_dividend_update(conn_params)
    #read_s_and_p_tickers_from_CSV(conn_params)
    #load_expiration_dates_all_tickers(conn_params)
    #store_nightly_interest_rate(conn_params)
    '''
    start_date = '2018-01-01'
    end_date_dt = dt.datetime.today().date()
    end_date = dt.datetime.strftime(end_date_dt, "%Y-%m-%d")
    iterate_through_S_and_P_store_dividend_yields(start_date, end_date, conn_params)'''
    store_interest_rates_in_db(conn_params)
    #initalize_options_table(conn_params)
    #iterate_through_S_and_P_store_dividends(start_date, end_date, conn_params)
    #iterate_through_S_and_P_store_stock_values(conn_params)
    #store_stock_price_history_yfinance('NVDA', conn_params )
    #store_stock_dividends_yfinance("AAPL","2016-01-01","2026-06-04", conn_params)
    #calculate_and_store_dividend_yields_database('NVDA', start_date, end_date, conn_params)
    #target_date = dt.datetime.strptime('2026-02-05', '%Y-%m-%d')
    #end_date = target_date + timedelta(days = 1)

    #store_stock_price_history_yfinance('AAPL',conn_params)
    #initalize_expirations_table()
    #initalize_options_table(conn_params)
    #initialize_general_market_data_table(conn_params)