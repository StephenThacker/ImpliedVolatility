import os
from dotenv import load_dotenv   # loads your existing .env file
import pytest
from testcontainers.postgres import PostgresContainer
import psycopg2
from psycopg2.extras import execute_values
from datetime import date
from datetime import timedelta

#db version: postgres:18-alpine

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")    
DB_NAME = os.getenv("DB_NAME")

if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
    raise ValueError("Missing one or more DB_ variables in your .env file")

REAL_DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

START_DATE = date(2026, 5, 11)
END_DATE   = date(2026, 5, 13)

def create_schema(conn_params: dict[str,str]):
    with psycopg2.connect(**conn_params) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                '''
                CREATE TABLE IF NOT EXISTS options (
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
                    );
                    
                    CREATE TABLE IF NOT EXISTS future_predictions (
                      date_of_creation DATE,
                      future_date DATE,
                      ticker VARCHAR(20),
                      estimated_dividend DOUBLE PRECISION DEFAULT 0.0,
                      
                      PRIMARY KEY (date_of_creation, future_date, ticker)
                      );


                CREATE TABLE IF NOT EXISTS stock_data (
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
                    );

                CREATE TABLE IF NOT EXISTS market_data (
                    date DATE,
                    risk_free_rate DOUBLE PRECISION,
                    s_and_p_tickers TEXT[],
                    s_and_p__master TEXT[] ,
                    s_and_p_changes TEXT[] ,
                    
                    PRIMARY KEY (date)
                );               
                ''')

            
    return

def copy_data_from_real_db(conn_params_test: dict[str, str], conn_params_real_url: str):

    with psycopg2.connect(**conn_params_test) as test_conn:
        with test_conn.cursor() as test_curs:
            with psycopg2.connect(conn_params_real_url) as real_conn:
                with real_conn.cursor() as real_curs:
                    
                    print("starting future predictions block")
                    #future predictions block
                    future_predictions_end_date = START_DATE + timedelta(days = 720)
                    future_predictions_real_query = ''' SELECT * FROM future_predictions WHERE future_date >= %s and future_date <= %s'''
                    future_pred_args = (START_DATE, future_predictions_end_date)
                    real_curs.execute(future_predictions_real_query, future_pred_args)
                    results = real_curs.fetchall()
                    print("pulled " , len(results), "from future divs")

                    
                    future_pred_insert_query = '''INSERT INTO future_predictions (date_of_creation,future_date, ticker, estimated_dividend)
                                                  VALUES %s ON CONFLICT (date_of_creation, future_date, ticker) DO NOTHING'''
                    if results:
                        execute_values(test_curs, future_pred_insert_query, results, page_size= 1000)
                        test_conn.commit()


                    print("starting options block")
                    #options_block
                    options_query = ''' SELECT ticker, expiration, strike, option_type, price_date, close, midpoint, bs_implied_vol, bin_imp_vol
                                        FROM options WHERE price_date >= %s AND price_date <= %s '''
                    
                    options_args = (START_DATE, END_DATE)
                    
                    real_curs.execute(options_query,options_args)
                    options_results = real_curs.fetchall()
                    print("pulled " , len(options_results), "from options")


                    options_insert_command = '''INSERT INTO options (ticker, expiration, strike, option_type, price_date, close, midpoint, 
                                                bs_implied_vol, bin_imp_vol) VALUES %s ON CONFLICT 
                                                (ticker, expiration, price_date, strike, option_type) DO NOTHING'''

                    if options_results:
                        execute_values(test_curs,options_insert_command, options_results, page_size= 1000)
                        test_conn.commit()

                    #stock_block
                    print("starting stock results block")
                    stock_query = '''SELECT ticker, date, dividend, close FROM stock_data WHERE date >= %s AND date <= %s'''
                    stock_args = (START_DATE, END_DATE)
                    real_curs.execute(stock_query, stock_args)
                    stock_results = real_curs.fetchall()
                    print("pulled " , len(stock_results), "from stock_results")
                    stock_insert_query = '''INSERT INTO stock_data (ticker ,date, dividend, close) VALUES %s ON CONFLICT (ticker,date) DO NOTHING'''

                    if stock_results:
                        execute_values(test_curs, stock_insert_query, stock_results, page_size = 1000)
                        test_conn.commit()

                    print("starting market_data block")
                    market_data_query = '''SELECT * FROM market_data'''
                    real_curs.execute(market_data_query)
                    market_results = real_curs.fetchall()
                    print("pulled " , len(market_results), "from market_results")

                    market_insert_query = '''INSERT INTO market_data (date, risk_free_rate, s_and_p_tickers,s_and_p__master, s_and_p_changes) VALUES 
                                             %s ON CONFLICT (date) DO NOTHING'''
                    
                    if market_results:
                        execute_values(test_curs, market_insert_query, market_results, page_size=1000)
                        test_conn.commit()


                    
                        


    return


def start_test_db():
    postgres = PostgresContainer('postgres:18-alpine')
    postgres.start()                  

    conn_params = {
        'host': postgres.get_container_host_ip(),
        'port': postgres.get_exposed_port(5432),
        'user': postgres.username,
        'password': postgres.password,
        'dbname': postgres.dbname,
    }

    create_schema(conn_params)
    copy_data_from_real_db(conn_params, REAL_DATABASE_URL)

    return conn_params, postgres


if __name__ == "__main__":
    print("Starting manual test container ...")
    
    with PostgresContainer('postgres:18-alpine') as postgres:
        conn_params = {
            'host': postgres.get_container_host_ip(),
            'port': postgres.get_exposed_port(5432),
            'user': postgres.username,
            'password': postgres.password,
            'dbname': postgres.dbname,
        }

        create_schema(conn_params)
        copy_data_from_real_db(conn_params, REAL_DATABASE_URL)
        print("Finished test")
