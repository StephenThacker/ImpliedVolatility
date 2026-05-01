from dotenv import load_dotenv
import os
from initialize_database import nightly_routine, store_nightly_interest_rate, scrape_finviz_for_dividend_data, iterate_through_S_and_P_store_dividend_yields
from implied_vol import theta_data_nightly_routine
import datetime as dt
from datetime import timedelta

load_dotenv()

if __name__ == "__main__":
    conn_params = {
        "host": "db",
        "database": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "port": "5432"
    }
    tickers = ['AAPL', 'PLTR', 'CVX']
    store_nightly_interest_rate(conn_params)
    print("interest rate routine finished")
    #scrape_finviz_for_dividend_data()
    print("dividend scrape finished")
