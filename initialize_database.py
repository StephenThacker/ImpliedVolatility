from dotenv import load_dotenv
import psycopg2
import os
import pandas as pd
import openpyxl
from io import StringIO

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



def initialize_stock_history_table():
    return


def initialize_stock_data_table(conn_params):
    create_table = '''CREATE TABLE IF NOT EXISTS stock_data (
       ticker VARCHAR(10),
       dividend_yield DOUBLE PRECISION,

       PRIMARY KEY (ticker)
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
    print("columns")
    print(df.columns)

    pandas_generator = df[cols].itertuples(index=False, name=None)

    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                pandas_generator = df[cols].itertuples(index = False,name = None )
                cur.executemany(insert_sql,pandas_generator)
    except Exception as e:
        print(e)              

    return




if __name__ == "__main__":
    conn_params = {
    "host": "localhost",
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": "5432"
    }
    store_interest_rates_in_db(conn_params)

    #initalize_expirations_table()
    #initalize_options_table()
    #initialize_general_market_data_table(conn_params)
    #initialize_stock_data_table(conn_params)

