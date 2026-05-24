import os
from dotenv import load_dotenv   # loads your existing .env file
import pytest
from testcontainers.postgres import PostgresContainer
import psycopg2
from psycopg2.extras import execute_values
from datetime import date

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

START_DATE = date(2025, 3, 10)
END_DATE   = date(2025, 3, 12)

def create_schema(conn_params: dict[str,str]):
    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            cur.execute(
                '''
                CREATE TABLE IF NOT EXISTS expiration_series (
                    ticker VARCHAR(10),
                    dates  DATE[] NOT NULL,
                    PRIMARY KEY (ticker));
                
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
                    S_and_P_tickers TEXT[],
                    S_and_P__master TEXT[] DEFAULT NONE,
                    S_and_P_changes TEXT[] DEFAULT NONE,
                    
                    PRIMARY KEY (date)
                );               
                ''')

            
    return

def copy_data_from_real_db(conn_params_test: dict[str,str], conn_params_real_url: str):

    return

@pytest.fixture(scope='session')
def postgres_container():
    with PostgresContainer('postgres:18-alpine') as postgres:
        conn_params = {
            'host': postgres.get_container_host_ip(),
            'port': postgres.get_exposed_port(5432),
            'user': postgres.username,
            'password': postgres.password,
            'dbname': postgres.dbname,
        }

        create_schema(conn_params)

        copy_data_from_real_db(conn_params)

        yield postgres






