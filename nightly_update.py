from dotenv import load_dotenv
import datetime as dt
import os
from initialize_database import store_nightly_interest_rate, iterate_through_S_and_P_store_dividend_yields
from implied_vol import thetadata_options_scrape_EOD
import datetime as dt
from datetime import timedelta
from polygonio_scrape import pull_div_data_poly_for_all

load_dotenv()

def nightly_update(start_date:dt.datetime, end_date:dt.datetime, conn_params = None, base_url = "http://host.docker.internal:25503/v3"):
    if conn_params == None:
        conn_params = {
        "host": "db",
        "database": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "port": "5432"}


    options_scrape = thetadata_options_scrape_EOD()
    store_nightly_interest_rate(conn_params)
    print("interest rate routine finished")
    options_scrape.scrape_stock_data_theta_data_S_and_P(start_date,end_date,conn_params, base_url= base_url)
    print("stock data scrape finished")
    options_scrape.scrape_options_data_theta_data_S_and_P(start_date,end_date,conn_params, base_url= base_url)
    print("options data scrape finished")
    dividend_results = pull_div_data_poly_for_all(conn_params, start_date, end_date)
    print("dividend pull successful")
    if not dividend_results:
        print("no new recorded tickers during this period")
    if dividend_results:
        for ticker in dividend_results:
            print("dividends recorded for " , ticker)
    if dividend_results:
        iterate_through_S_and_P_store_dividend_yields(start_date, end_date, conn_params)
        print("dividends stored")
    
    options_scrape.build_options_surface_entire_S_and_P(conn_params, start_date, end_date, 'Black Scholes')
    print("completed Black Scholes")
    options_scrape.build_options_surface_entire_S_and_P(conn_params, start_date, end_date, 'Binomial Tree')
    print("completed Binomial Tree")


if __name__ == "__main__":

    today = dt.datetime.today()
    one_week_ago = today - timedelta(days= 20)


    nightly_update(one_week_ago, today)
