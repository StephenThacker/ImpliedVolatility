from dotenv import load_dotenv
import os
from initialize_database import nightly_routine, store_nightly_interest_rate
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
    #nightly_routine(conn_params)
    #theta_data_nightly_routine(tickers)
