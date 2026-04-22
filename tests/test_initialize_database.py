import unittest
import yfinance as yf
import pandas
import os
from dotenv import load_dotenv
import holidays
import datetime as dt
from datetime import date, timedelta
import psycopg2
import math



load_dotenv()


class Testinitialize_database(unittest.TestCase):
    
    
    def test_stock_data_accuracy(self):
        conn_params = {
            "host": "localhost",
            "database": os.getenv("DB_NAME"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "port": "5432"
            }
        

        ticker = 'AAPL'
        stock = yf.Ticker(ticker)
        history = stock.history(period = '5d')

        #getting previous day (because market needs to be closed)
        today = dt.datetime.today()
        yesterday = today - timedelta(days=1)
        yesterday = yesterday.date()

        if yesterday.weekday() >= 5 or yesterday in holidays.financial_holidays('NYSE'):
            self.fail("Market closed/holiday")

        specific_day_row = history.loc[yesterday.strftime('%Y-%m-%d')]
        stock_close = specific_day_row['Close']        

        close_query = '''SELECT close, div_yield_per FROM stock_data where date = %s AND ticker = %s'''


        args = [yesterday, ticker]

        try:
            with psycopg2.connect(**conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute(close_query, args)
                    results = cur.fetchall()
                    stock_close_datab = results[0][0]
                    div_yield = results[0][1]
                    
        except Exception as e:
            print(e) 

        #div yield test is sub-optimal. div yield changes daily and compares previous day div yield @ close to current &
        # div yield attribute not widely supported for most tickers on yfinance. But, don't know better easy source of div yield data currently
        if yf.Ticker(ticker).info.get('dividendYield') != None:
            yf_div_y = yf.Ticker(ticker).info.get('dividendYield')/100
        else:
            self.fail("No dividend yield")
        #Check stock values
        with self.subTest(check = 'close_price'):
            self.assertAlmostEqual(stock_close,stock_close_datab,delta=0.1)
        with self.subTest(check = "dividend yield"):
            self.assertTrue(math.isclose(yf_div_y,div_yield,rel_tol=0.3))






if __name__ == "__main__":
    conn_params = {
        "host": "localhost",
        "database": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "port": "5432"
        }
    
    #example = Testinitialize_database()
    #example.test_stock_data_accuracy()
    
    unittest.main()