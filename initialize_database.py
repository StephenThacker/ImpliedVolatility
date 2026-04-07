from dotenv import load_dotenv
import psycopg2
import os
import pandas as pd
import openpyxl
from io import StringIO
import yfinance as yf
from datetime import date, timedelta
import datetime as dt

#Contains "one-time load" scripts related to initalizing database tables or transferring CSV/API historical data to databases
#i.e., pull interest rates from CSV file and store in database

load_dotenv()


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
        r'C:\Users\steph\Desktop\Coding\Github Quant Finance\CSV_FILES\Interest_Rates_DB.xlsx',
        sheet_name='Results')
    print("df columns 1")
    print(df.columns)

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
    except Exception as e:
        print(e)              

    return

#stores stock dividends from date range into database
def store_stock_dividends_yfinance(ticker, date_start,date_end, conn_params):
    stock = yf.Ticker(ticker)

    dividends = stock.dividends
    dividends = dividends[date_start:date_end]
    data = {'ticker' : ticker, "date" :dividends.index.date,'dividend' : dividends.values}
    df = pd.DataFrame(data)
    print(df["date"])
    del dividends
    insert_sql = '''INSERT INTO stock_data (ticker, date, dividend) VALUES (%s, %s, %s) 
                    ON CONFLICT DO NOTHING'''
    

    
    pandas_generator = df.itertuples(index= False, name = None)
    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                cur.executemany(insert_sql,pandas_generator)
    except Exception as e:
        print(e)
        
    """
    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM stock_data")
                results = cur.fetchall()
    except Exception as e: 
        print(e)

    for row in results:
        print(row)



    #dividends = stock.dividends.loc[date_start,date_end]"""

    #stores EOD stock price history for last 10 years for a single stock ticker
def store_stock_price_history_yfinance(ticker, conn_params):

    stock = yf.Ticker(ticker)

    stock_price = stock.history(period="10y")


    df_data = {"ticker": ticker, "date": stock_price.index.date, "close": stock_price["Close"], "open": stock_price["Open"], \
                "high": stock_price["High"],"low": stock_price["Low"], "volume": stock_price["Volume"]}
    
    df = pd.DataFrame(df_data)



    insert_sql = '''INSERT INTO stock_data 
                    (ticker, date, close, open, high, low, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING'''
    
    pandas_generator = df.itertuples(index = False, name = None)


    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                cur.executemany(insert_sql,pandas_generator)
    except Exception as e:
        print(e)

    '''
    query_first = "SELECT * FROM stock_data ORDER BY date ASC LIMIT 100;"
    query_last = "SELECT * FROM stock_data ORDER BY date DESC LIMIT 100;"

    try:
            with psycopg2.connect(**conn_params) as conn:
                df_first = pd.read_sql_query(query_first, conn)
                
                df_last = pd.read_sql_query(query_last, conn)

                print(df_first)
                
                print(df_last.sort_values('date'))

    except Exception as e:
        print(e)'''




    return




if __name__ == "__main__":
    conn_params = {
    "host": "localhost",
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": "5432"
    }
    #store_interest_rates_in_db(conn_params)
    #initialize_stock_data_table(conn_params)
    #store_stock_dividends_yfinance("AAPL","2016-01-01","2026-06-04", conn_params)
    #target_date = dt.datetime.strptime('2026-02-05', '%Y-%m-%d')
    #end_date = target_date + timedelta(days = 1)

    #store_stock_price_history_yfinance('AAPL',conn_params)
    #initalize_expirations_table()
    initalize_options_table(conn_params)
    #initialize_general_market_data_table(conn_params)

