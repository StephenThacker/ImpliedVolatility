from dotenv import load_dotenv
import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf
from scipy.optimize import brentq
import plotly.graph_objects as go
import datetime as dt
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from psycopg2.extras import execute_values
import os
import psycopg2
import time
from numba import njit, prange
import httpx
import io
from datetime import date, timedelta
import csv
import holidays
from utils import get_S_and_P_composite
import asyncio
from collections.abc import Iterator
import plotly
from tests import conftests
import implied_vol
from testcontainers.postgres import PostgresContainer
from data_helpers.ephemeral_db import start_test_db


#need try, finally code block here for database management of ephemeral database.


class binomial_tree_vellekoop:

    def __init__(self):
        pass


if __name__ == "__main__":
    pass