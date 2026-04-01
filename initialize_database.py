from dotenv import load_dotenv
import psycopg2
import os

load_dotenv()


def main():
    conn_params = {
    "host": "localhost",
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": "5432"
    }

    SQL_QUERY = """ CREATE TABLE IF NOT EXISTS options (
                    ticker VARCHAR(10),
                    symbol VARCHAR(50),
                    expiration DATE,
                    strike DOUBLE PRECISION,
                    option_type VARCHAR(5),
                    created TIMESTAMP,
                    price_date DATE,
                    last_trade TIMESTAMPTZ,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume BIGINT,
                    count BIGINT,
                    bid_size BIGINT,
                    bid_exchange BIGINT,
                    bid DOUBLE PRECISION,
                    bid_condition BIGINT,
                    ask_size BIGINT,
                    ask_exchange BIGINT,
                    ask DOUBLE PRECISION,
                    ask_condition BIGINT,


                    PRIMARY KEY (ticker, expiration, price_date, strike, option_type)
                    )

                    """

    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(SQL_QUERY)
                conn.commit()

                
    except Exception as e:
        print(f"DB Error: {e}")





if __name__ == "__main__":


    main()