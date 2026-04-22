import unittest
import yfinance as yf
import pandas
import os
from dotenv import load_dotenv
import holidays
import datetime as dt
from datetime import date, timedelta
import psycopg2



load_dotenv()


class Testinitialize_database(unittest.TestCase):
    
    
    def test_stock_close_data_accuracy(self):
        conn_params = {
            "host": "localhost",
            "database": os.getenv("DB_NAME"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "port": "5432"
            }
        

        ticker = 'AAPL'
        stock = yf.Ticker(ticker)
        print(type(stock))
        history = stock.history(period = '5d')

        #getting previous day (because market needs to be closed)
        today = dt.datetime.today()
        yesterday = today - timedelta(days=1)
        yesterday = yesterday.date()

        if yesterday.weekday() >= 5 or yesterday in holidays.financial_holidays('NYSE'):
            self.fail("Market closed/holiday")

        specific_day_row = history.loc[yesterday.strftime('%Y-%m-%d')]
        stock_close = specific_day_row['Close']        

        SQL_query = '''SELECT close FROM stock_data where date = %s AND ticker = %s'''


        args = [yesterday, ticker]

        try:
            with psycopg2.connect(**conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute(SQL_query, args)
                    results = cur.fetchall()
                    
        except Exception as e:
            print(e) 

        database_value = results[0][0]

        #Check stock values
        
        self.assertAlmostEqual(stock_close,database_value,delta=0.1)



if __name__ == "__main__":
    conn_params = {
        "host": "localhost",
        "database": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "port": "5432"
        }
    
    example = Testinitialize_database()
    example.test_check_data_accuracy()