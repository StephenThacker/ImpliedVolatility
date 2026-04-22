import streamlit as st
import datetime as dt
from implied_vol import thetadata_options_scrape_EOD
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Implied Volatility Surface Animation", layout="wide")
st.title("Implied Volatility Surface Animation")

with st.sidebar:
    st.header("Controls")
    
    tickers_input = st.text_input("Tickers ", "PLTR")
    ticker_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start Date", dt.date(2026, 1, 15))
    end_date = col2.date_input("End Date", dt.date(2026, 4, 10))
    
    option_types = st.multiselect("Option Type(s)", ["CALL", "PUT"], default=["PUT"])
    calc_type = st.selectbox("Calculation Method", ["Binomial Tree", "Black Scholes"])
    
    low_coef = st.slider("Lower Strike × Spot", 0.5, 1.0, 0.7, 0.05)
    high_coef = st.slider("Upper Strike × Spot", 1.0, 2.0, 1.3, 0.05)
    
    interp = st.selectbox("Interpolation", ["linear", "cubic", "nearest"], index=0)

if st.button("Generate Animation(s)", type="primary", use_container_width=True):
    if not ticker_list:
        st.error("Please enter at least one ticker")
    else:
        theta = thetadata_options_scrape_EOD()
        conn_params = {
            "host": "localhost",
            "database": os.getenv("DB_NAME"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "port": "5432"
        }
        
        # One tab per ticker
        tabs = st.tabs(ticker_list)
        
        for i, ticker in enumerate(ticker_list):
            with tabs[i]:
                st.subheader(ticker)
                
                # Side-by-side for CALL and PUT (best for comparison)
                if len(option_types) == 2:
                    col_call, col_put = st.columns(2)
                    cols = {"CALL": col_call, "PUT": col_put}
                else:
                    cols = {opt: st.container() for opt in option_types}
                
                for opt_type in option_types:
                    with cols.get(opt_type, st.container()):
                        st.caption(f"{opt_type} options")
                        with st.spinner(f"Building {opt_type} animation..."):
                            fig = theta.build_options_animation(
                                ticker=ticker,
                                start_date=dt.datetime.combine(start_date, dt.datetime.min.time()),
                                end_date=dt.datetime.combine(end_date, dt.datetime.min.time()),
                                low_strike_coef=low_coef,
                                high_str_coef=high_coef,
                                interp_method=interp,
                                option_type=opt_type,
                                conn_params=conn_params,
                                calculation_type=calc_type
                            )
                            
                            st.plotly_chart(
                                fig,
                                use_container_width=True,
                                height=800,          # much bigger charts
                                config={"scrollZoom": True, "displayModeBar": True}
                            )
                            
                            st.download_button(
                                label=f"Download {opt_type} as HTML",
                                data=fig.to_html(full_html=True),
                                file_name=f"iv_animation_{ticker}_{opt_type}_{start_date}_{end_date}.html",
                                mime="text/html",
                                key=f"download_{ticker}_{opt_type}"
                            )