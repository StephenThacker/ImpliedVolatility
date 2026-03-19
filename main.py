import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

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
    main()