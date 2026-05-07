import pandas as pd
import psycopg2
import datetime as dt
import os
from dotenv import load_dotenv
from datetime import timedelta

load_dotenv()

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


#Takes date and calculates what the S_and_P was on that date
def get_S_and_P_for_date(conn_params: dict[str,str], date: dt.datetime) -> list[str]:

    date = date.date()

    sql_master_query = '''SELECT date, s_and_p__master FROM market_data WHERE s_and_p__master IS NOT NULL ORDER BY date DESC LIMIT 1'''

    sql_changes_query = '''SELECT date, s_and_p_changes FROM market_data WHERE s_and_p_changes IS NOT NULL ORDER BY date DESC'''

    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(sql_master_query)
                results = cur.fetchall()
            df = pd.read_sql_query(sql_changes_query, conn)
    except Exception as e:
        print("issue with collecting S_and_P list", e)
    
    master_row = results[0]
    master_date = master_row[0]
    S_and_P_master = master_row[1]
    S_and_P_master = S_and_P_master[:]

    if master_date == date:
        return S_and_P_master
    #date after master
    if master_date < date:
        df = df[(df['date'] > master_date) & (df['date']<= date)]
    if master_date > date:
        df = df[(df['date'] < master_date) & (df['date'] >= date)]
                
    df_iteritems = df.itertuples(index = False,name = None)

    additions = []
    subtractions = []
    for row in df_iteritems:
        date_change,ticker_list = row
        for ticker in ticker_list:
            if ticker[0] == "+":
                if date >= date_change:
                    additions.append(ticker[1:])
                if date < date_change:
                    subtractions.append(ticker[1:])
            if ticker[0] == '-':
                if date >= date_change:
                    subtractions.append(ticker[1:])
                if date < date_change:
                    additions.append(ticker[1:])
    
    modified_list = S_and_P_master[:]
    #iterate through S_and_P_master and update list
    for ticker in additions:
        if ticker not in S_and_P_master[:]:
            modified_list.append(ticker)
    
    for ticker in subtractions:
        if ticker in S_and_P_master[:]:
            modified_list.remove(ticker)

    return modified_list


if __name__ == "__main__":
    conn_params = {
    "host": "localhost",
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": "5432"
    }
    test_date_1 = dt.datetime.today() - timedelta(days=30)
    test_date_2 = dt.datetime.today()

    print("checking first one")
    test_1 = get_S_and_P_for_date(conn_params,test_date_1)
    if 'CTRA' in test_1:
        print('CTRA',' passed')
    if 'CTRA' not in test_1:
        print('CTRA', "failed")
    
    if 'VEEV' in test_1:
        print("Veev failed")
    else:
        print('VEEV passed')


    print("checking second one")
    test_2 =  get_S_and_P_for_date(conn_params,test_date_2)


    if 'CTRA' in test_2:
        print('CTRA',' failed')
    if 'CTRA' not in test_2:
        print('CTRA', "passed")
    
    if 'VEEV' in test_2:
        print("Veev passed")
    else:
        print('VEEV failed')