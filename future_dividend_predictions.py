import pandas as pd
import psycopg2
import datetime as dt
import os
from dotenv import load_dotenv
from datetime import timedelta
from utils import get_S_and_P_composite


def iterate_composite_tickers_dividend_prediction(conn_params, ticker: str):
    start_date = dt.datetime.today() - timedelta(days=2000)
    end_date = dt.datetime.today()
    tickers = get_S_and_P_composite(conn_params,start_date, end_date)

    for ticker in tickers:
        generate_future_dividend_predictions(conn_params, ticker)
    return

def pull_last_dividend_date(conn_params, ticker:str):
    start_date = dt.datetime.today() - timedelta(days=95)
    end_date = dt.datetime.today() + timedelta(days = 95)

    sql_query = '''SELECT date, dividend FROM stock_data WHERE date >= %s AND date <= %s AND ticker = %s
                   AND dividend > 0 ORDER BY date DESC LIMIT 1'''
    
    args = [start_date, end_date, ticker]
    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            cur.execute(sql_query, args)
            results = cur.fetchall()
    
    if not results:
        return []

    else:
        return results[0]



#Not all S&P 500 companies pay quarterly dividends. Some pay non-quarterly dividends
def generate_future_dividend_predictions(conn_params, ticker: str):

    div_pull = pull_last_dividend_date(conn_params, ticker)
    if not div_pull:
        print("No dividends in this range")
        return []
    date, last_div = div_pull

    insert_query = '''INSERT INTO future_predictions (date_of_creation, future_date, ticker, estimated_dividend)
                      VALUES (%s, %s, %s, %s) ON CONFLICT (date_of_creation,future_date, ticker)
                      DO UPDATE SET
                      estimated_dividend = EXCLUDED.estimated_dividend
                      '''

    today = dt.datetime.today()
    #Can upgrade this method to have better dividend predictions
    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            date_var = date
            date_fin = date_var + timedelta(days= 900)
            while date_var <= date_fin:
                date_var = date_var + timedelta(days=91)
                args = [today, date_var,ticker, last_div]
                cur.execute(insert_query, args)
        conn.commit()


    #Manual test code
    Test_query = '''SELECT * FROM future_predictions WHERE ticker = %s'''
    args = [ticker]

    results = []
    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            cur.execute(Test_query, args)
            results = cur.fetchall()

    for row in results:
        print(row)

    return


if __name__ == "__main__":
    load_dotenv()

    conn_params = {
    "host": "localhost",
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": "5432"
    }
    iterate_composite_tickers_dividend_prediction(conn_params, 'AAPL')

