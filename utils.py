import pandas as pd
import psycopg2

def S_and_P_tickers(conn_params):

    ticker_query = '''SELECT S_and_P_tickers FROM market_data WHERE S_and_P_tickers IS NOT NULL 
                      ORDER By date DESC 
                      LIMIT 1'''

    try:
        with psycopg2.connect(**conn_params) as conn:
            df = pd.read_sql_query(ticker_query, conn)

    except Exception as e:
        print(e)
    
    tickers = df.iloc[-1].tolist()[0]
    return tickers