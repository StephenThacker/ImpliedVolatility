import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
import datetime as dt
import plotly.graph_objects as go
from scipy.interpolate import griddata


class options_chain:
    
    def __init__(self, ticker):
        self.ticker = ticker
        self.options_chains_dict = {}
        self.fetch_options_data(self.ticker)

    def vega(self, stock_price, time_to_expiration, cont_div_yield, d_1):
        vega = stock_price*np.exp(-1*cont_div_yield*time_to_expiration)*1/np.sqrt(2*np.pi)*np.exp(-1*(d_1**2)/2)*np.sqrt(time_to_expiration)
        return vega

    def black_scholes_call_option(self, stock_price, contin_div_yield, time_to_expiration,d_1, d_2, strike_price , risk_free_rate):
        price = stock_price*np.exp(-contin_div_yield*time_to_expiration)*norm.cdf(d_1) - strike_price*np.exp(-risk_free_rate*time_to_expiration)*norm.cdf(d_2)
        return price

    def d_1(self, strike_price, stock_price, risk_free_rate, cont_div_yield, sigma, time_to_expiration):
        d_1 = (np.log(stock_price/strike_price)+(risk_free_rate - cont_div_yield + (sigma**2)/2)*time_to_expiration)/(sigma*np.sqrt(time_to_expiration))
        return d_1

    def d_2(self, d_1, time_to_expiration, sigma):
        d2 = d_1 - sigma*np.sqrt(time_to_expiration)
        return d2
    
    def newton_raphson_method(self, epsilon,stock_price, option_price,risk_free_rate, contin_div_yield, time_to_expiration, strike_price):
        sigma = 0.2
        black_scholes_cost = 1
        market_cost = 0
        max_iter = 400
        count = 0
        while (np.abs(black_scholes_cost - market_cost) > epsilon) and count< max_iter:

            d_1 = self.d_1(strike_price, stock_price, risk_free_rate,contin_div_yield,sigma,time_to_expiration)
            d_2 = self.d_2(d_1, time_to_expiration, sigma)
            black_scholes_cost = self.black_scholes_call_option(stock_price, contin_div_yield, time_to_expiration, d_1, d_2, strike_price, risk_free_rate)
            market_cost = option_price
            vega = self.vega(stock_price, time_to_expiration, contin_div_yield, d_1)

            if abs(vega) < 1e-8:
                break
            sigma = sigma - (black_scholes_cost - market_cost)/(vega)

            count += 1
        return sigma
    
    def dates_to_expiration(self, expiration_date):
        expiration_date = dt.datetime.strptime(expiration_date, "%Y-%m-%d").date()
        days_to_expiration = (expiration_date - dt.date.today()).days
        fraction_of_days = days_to_expiration/365
        return fraction_of_days
    
    def num_dates_to_expir(self,expiration_date):
        expiration_date = dt.datetime.strptime(expiration_date, "%Y-%m-%d").date()
        days_to_expiration = (expiration_date - dt.date.today()).days

        return days_to_expiration

    def fetch_options_data(self, ticker):
        self.ticker = yf.Ticker(ticker)
        expiration_dates = self.ticker.options
        options = self.ticker.option_chain(expiration_dates[2])
        self.calls = options.calls
        self.info = self.ticker.info
        self.options_chains_dict = {exp : self.ticker.option_chain(exp).calls for exp in expiration_dates}
        last_stock_price = self.info.get('regularMarketPrice') or ticker.history(period="1d")['Close'][0]
        risk_free = yf.Ticker("^IRX").history(period="1d")['Close'][-1]/100
        if yf.Ticker(ticker).info.get('dividendYield') != None:
            dividend_yield = yf.Ticker(ticker).info.get('dividendYield')/100
        else:
            dividend_yield = 0

        for key in self.options_chains_dict.keys():
            date_fraction = self.dates_to_expiration(key)
            self.options_chains_dict[key]['calcImpliedVol'] = np.vectorize(self.newton_raphson_method)(1e-5, last_stock_price,self.options_chains_dict[key]['lastPrice'],risk_free,dividend_yield,date_fraction,self.options_chains_dict[key]['strike'])
            self.options_chains_dict[key]['daystoExpir'] = np.vectorize(self.num_dates_to_expir)(key)
            print(self.options_chains_dict[key].columns)
            print(self.options_chains_dict[key].head())
            print(self.options_chains_dict[key]['calcImpliedVol'])
            print(self.options_chains_dict[key]['daystoExpir'])

        return
    
    def plot_imp_vol_curve(self):
        x = np.linspace(-5,5, 100)
        y = np.linspace(-5,5,100)
        X,Y = np.meshgrid(x,y)


    def plot_imp_vol_surface(self):
        import plotly.graph_objects as go
        from scipy.interpolate import griddata

        last_price = self.info.get('regularMarketPrice')
        if last_price is None:
            last_price = yf.Ticker(self.ticker.ticker).history(period="1d")['Close'][0]

        # ±15% of ATM price now
        lower_strike = last_price * 0.85
        upper_strike = last_price * 1.15

        max_days_to_exp = 60

        strikes_all = []
        maturities_all = []
        implied_vols_all = []

        for exp_date, df in self.options_chains_dict.items():
            days_to_exp = self.num_dates_to_expir(exp_date)
            if days_to_exp > max_days_to_exp or days_to_exp < 0:
                continue

            valid_mask = (~df['calcImpliedVol'].isna()) & \
                        (df['calcImpliedVol'] > 0) & \
                        (df['calcImpliedVol'] <= 2.0) & \
                        (df['strike'] >= lower_strike) & (df['strike'] <= upper_strike)

            strikes = df.loc[valid_mask, 'strike'].values
            maturity = days_to_exp
            implied_vols = df.loc[valid_mask, 'calcImpliedVol'].values

            if len(strikes) == 0:
                continue

            strikes_all.extend(strikes)
            maturities_all.extend([maturity] * len(strikes))
            implied_vols_all.extend(implied_vols)

        strikes_all = np.array(strikes_all)
        maturities_all = np.array(maturities_all)
        implied_vols_all = np.array(implied_vols_all)

        if len(strikes_all) == 0:
            print("No data points available for plotting after filtering.")
            return

        strike_grid = np.linspace(strikes_all.min(), strikes_all.max(), 50)
        maturity_grid = np.linspace(maturities_all.min(), maturities_all.max(), 50)
        M_grid, K_grid = np.meshgrid(maturity_grid, strike_grid)

        IV_grid = griddata(
            points=(maturities_all, strikes_all),
            values=implied_vols_all,
            xi=(M_grid, K_grid),
            method='linear'
        )

        fig = go.Figure(data=[go.Surface(
            x=M_grid,
            y=K_grid,
            z=IV_grid,
            colorscale='Viridis',
            colorbar=dict(title='Implied Volatility')
        )])

        fig.update_layout(
            title=f"Implied Volatility Surface for {self.ticker.ticker} (Filtered)",
            scene=dict(
                xaxis_title='Days to Expiration',
                yaxis_title='Strike Price',
                zaxis_title='Implied Volatility',
            ),
            autosize=True,
            width=800,
            height=700
        )

        fig.show()






def main():
    nvda_options = options_chain("TSLA")
    nvda_options.plot_imp_vol_surface()

if __name__ == "__main__":
    main()
