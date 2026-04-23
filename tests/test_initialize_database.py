import unittest
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.insert(0,parent_dir)
import utils
import yfinance as yf
import pandas
import os
from dotenv import load_dotenv
import holidays
import datetime as dt
from datetime import date, timedelta
import psycopg2
import math
from bs4 import BeautifulSoup
import requests
import utils
import time



load_dotenv()

def get_dividend_yield(ticker: str) -> str:
    url = f"https://finviz.com/quote.ashx?t={ticker.upper()}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/128.0 Safari/537.36"
    }
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    div_cell = soup.find("td", string="Dividend TTM")
    try:
        if div_cell:
            yield_cell = div_cell.find_next_sibling("td")
            if yield_cell:
                text_var = yield_cell.get_text(strip=True)
                text_var = text_var.split('(')[1].split(')')[0]   # get '0.38%'
                text_var = float(text_var.strip('%'))
                return text_var
        
        return 0
    except Exception as e:
        return 0


class Testinitialize_database(unittest.TestCase):
    
    
    def test_stock_data_accuracy(self):
        conn_params = {
            "host": "localhost",
            "database": os.getenv("DB_NAME"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "port": "5432"
            }
        
        tickers = utils.S_and_P_tickers(conn_params)
        for ticker in tickers:
            with self.subTest(ticker=ticker):
                try: 
                    time.sleep(2)
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


                    t_div_yield = float(get_dividend_yield(ticker))/100
                    #Check stock values
                    with self.subTest(check = 'close_price'):
                        self.assertAlmostEqual(stock_close,stock_close_datab,delta=0.1)
                    with self.subTest(check = "dividend yield"):
                        self.assertTrue(math.isclose(t_div_yield,div_yield,rel_tol=0.25))
                except Exception as e:
                    self.fail(f"Unexpected crash for {ticker}: {e}")






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