from dotenv import load_dotenv
import os
from initialize_database import nightly_routine

load_dotenv()


if __name__ == "__main__":
    conn_params = {
        "host": "db",
        "database": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "port": "5432"
    }
    nightly_routine(conn_params)
