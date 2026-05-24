from dotenv import load_dotenv
import datetime as dt
from datetime import timedelta
import os
from nightly_update import nightly_update


load_dotenv()

def iterate_range(start_date:dt.datetime, end_date:dt.datetime,conn_params, batch_chunk_size: int = 15,  base_url = "http://127.0.0.1:25503/v3" ):
    
    date_var = start_date
    while date_var <= end_date:
        end_var = date_var + timedelta(days=batch_chunk_size)
        try:
            nightly_update(date_var,  end_var, conn_params, base_url)
        except Exception as e:
            print("error", e, " occurred during the dates: ", date_var, "and ", end_var)
        date_var = end_var
    
    #pick up last few days, when it's less than batch chunk size.
    date_var = date_var - timedelta(days=batch_chunk_size)
    nightly_update(date_var, end_date, conn_params, base_url)

    return


if __name__ == "__main__":
    conn_params = {
    "host": "localhost",
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": "5432"
    }

    end_date = dt.datetime.today() - timedelta(days=1)
    start_date = end_date - timedelta(days= 10)

    iterate_range(start_date, end_date,conn_params,15)

