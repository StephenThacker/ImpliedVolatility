from dotenv import load_dotenv
import os
from initialize_database import store_nightly_interest_rate, iterate_through_S_and_P_store_dividend_yields
from implied_vol import thetadata_options_scrape_EOD
import datetime as dt
from datetime import timedelta
from polygonio_scrape import pull_div_data_poly_for_all

load_dotenv()

if __name__ == "__main__":
    conn_params = {
        "host": "db",
        "database": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "port": "5432"
    }
    today = dt.datetime.today()
    one_week_ago = today - timedelta(days= 20)

    options_scrape = thetadata_options_scrape_EOD()

    store_nightly_interest_rate(conn_params)
    print("interest rate routine finished")
    options_scrape.scrape_stock_data_theta_data_S_and_P(one_week_ago,today,conn_params, base_url= "http://host.docker.internal:25503/v3")
    print("stock data scrape finished")
    options_scrape.scrape_options_data_theta_data_S_and_P(one_week_ago,today,conn_params, base_url= "http://host.docker.internal:25503/v3")
    print("options data scrape finished")
    dividend_results = pull_div_data_poly_for_all(conn_params, one_week_ago, today)
    print("dividend pull successful")
    if not dividend_results:
        print("no new recorded tickers during this period")
    if dividend_results:
        for ticker in dividend_results:
            print("dividends recorded for " , ticker)
    if dividend_results:
        iterate_through_S_and_P_store_dividend_yields(one_week_ago, today, conn_params)
        print("dividends stored")

