import os
import psycopg2
from dotenv import load_dotenv
import httpx
import io
from datetime import datetime, date, timedelta
import csv

load_dotenv()

def stream_stock_data_theta(today, ticker):

    BASE_URL = "http://127.0.0.1:25503/v3"

    PARAMS = {'symbol': ticker , "start_date" : date, "end_date" : date}

    print("today",today)
    start_date = today - timedelta(days = 3)
    end_date = today - timedelta(days = 3)

    if start_date.weekday() < 5:
        start_date = datetime.strftime(start_date,"%Y%m%d") 
        end_date = datetime.strftime(end_date,"%Y%m%d") 

        PARAMS['start_date'] = start_date
        PARAMS['end_date'] = end_date

        url = BASE_URL + '/stock/history/eod'

        with httpx.stream("GET", url, params = PARAMS, timeout= 60) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                for row in csv.reader(io.StringIO(line)):
                    print(row)

    else:
        return
    

    return

def main():
    conn_params = {
        "host": "localhost",
        "database": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "port": "5432"
    }

    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE TABLE IF NOT EXISTS hello_world (id SERIAL PRIMARY KEY, message TEXT);")
                cur.execute("INSERT INTO hello_world (message) VALUES (%s)", ("Hello!",))
                cur.execute("SELECT message FROM hello_world ORDER BY id DESC LIMIT 1;")
                print(f"Result: {cur.fetchone()[0]}")
                
                conn.commit()
    except Exception as e:
        print(f"DB Error: {e}")


if __name__ == "__main__":

    today_date = date.today()
    ticker = "AAPL"
    stream_stock_data_theta(today_date,ticker)

    main()