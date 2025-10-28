import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
import datetime as dt
import plotly.graph_objects as go
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

class Node:
    def __init__(self, node_stock_price):
        self.node_stock_price = node_stock_price
        self.value = None
        self.option_value = None
        self.up = None  
        self.down = None
        self.node_expected_price_value = None
        self.current_time = 0
        self.node_layer = 1
        self.node_number = 1
        #contains a list of all nodes that are connecting to this node on the upstream side of the graph. 
        # Each node is represented as a list, which countains a touple and a string
        #The touple gives the coordinates of the node, the string gives the direction of the branch that the prior node is emitting. ("i.e., up, or down")
        self.backwards_nodes = []

class binomial_tree:

    def __init__(self, strike_price, number_of_layers, initial_stock_price, initial_sigma, interest_rate, time_to_expiration, stock_dividend):
        self.strike_price = strike_price
        self.number_of_layers = number_of_layers
        self.root = Node(initial_stock_price)
        self.sigma = initial_sigma
        self.time_to_expiration = time_to_expiration
        self.interest_rate = interest_rate
        self.delta_t = None
        self.dividend = stock_dividend
        self.node_list = []
        self.define_time_segment()
        self.final_option_value = self.create_tree_structure()
        #print(self.final_option_value)

    def calculate_probability(self):
        try:
            self.probability = (np.exp((self.interest_rate-self.dividend) * self.delta_t) - self.d) / (self.u - self.d)
        except ZeroDivisionError:
            raise ValueError("Division by zero in probability calculation (u == d).")
        return
    
    def define_time_segment(self):
        self.time_to_expiration = self.time_to_expiration/365
        try:
            self.delta_t = self.time_to_expiration / (self.number_of_layers -1)
        except ZeroDivisionError:
            raise ValueError
        
        self.u = np.exp(self.sigma * np.sqrt(self.delta_t))
        self.d = np.exp(-self.sigma * np.sqrt(self.delta_t))
        return self.delta_t
    
    def create_tree_nodes(self):
        for i in range(self.number_of_layers):
            sub_list_array = [Node(None) for j in range(i + 1)]
            self.node_list.append(sub_list_array)
        return
    
    def calculate_all_node_stock_values(self):
        # Set the root layer price from self.root (assumes node_list[0][0] is root level)
        if self.node_list and self.node_list[0]:
            self.node_list[0][0].node_stock_price = self.root.node_stock_price

        # For each graph layer
        for i in range(1, self.number_of_layers):
            # Iterate across all nodes in layer
            for k in range(len(self.node_list[i])):
                current_node = self.node_list[i][k]
                backwards_nodes_list = current_node.backwards_nodes
                prices = []  # Collect computed prices from all parents
                for m, backward in enumerate(backwards_nodes_list):
                    node_tuple, node_direction = backward
                    node_tuple_x, node_tuple_y = node_tuple
                    try:
                        parent_node = self.node_list[node_tuple_x][node_tuple_y]
                        if node_direction == "up":
                            computed_price = parent_node.node_stock_price * self.u
                        elif node_direction == "down":
                            computed_price = parent_node.node_stock_price * self.d
                        else:
                            continue  # Skip unknown directions
                        prices.append(computed_price)
                    except (IndexError, AttributeError) as e:
                        print(f"Error at layer {i}, node {k}, parent {m}: {e}")
                        continue
                
                # Assign average price (handles recombination; single parent = the price itself)
                if prices:
                    current_node.node_stock_price = np.mean(prices)
                else:
                    current_node.node_stock_price = 0.0  # Fallback for no parents
                
       
        return
    
   
    def create_tree_structure(self):
        self.create_tree_nodes()
        self.create_recombining_tree_branches()
        self.calculate_all_node_stock_values()
        self.calculate_probability()
        return self.determine_option_value()

    
    def determine_option_value(self):
        #assigns values of options for the last layer of the tree
        last_layer_index = self.number_of_layers - 1
        for node in self.node_list[last_layer_index]:
            difference = node.node_stock_price - self.strike_price
            node.option_value = np.maximum(difference,0)

        count = last_layer_index - 1
        while count >= 0:
            for node in self.node_list[count]:
                up_node = node.up.option_value
                down_node = node.down.option_value
                node.option_value = self.assign_node_a_value(up_node, down_node)
            count = count - 1

        return self.node_list[0][0].option_value
        
    
    def assign_node_a_value(self, node_value_up, node_value_down):
        node_value = (np.exp(-1*(self.interest_rate- self.dividend)*self.delta_t)*(self.probability*node_value_up + (1 -self.probability)*node_value_down))
        return node_value      
    
    def create_recombining_tree_branches(self):
        count = len(self.node_list) -1
        while 0 <= count -1:
            for j in range(0,len(self.node_list[count -1])):
                self.node_list[count-1][j].up = self.node_list[count][j]
                self.node_list[count -1][j].down = self.node_list[count][j+1]
                self.node_list[count][j].backwards_nodes.append([(count-1,j),"up"])
                self.node_list[count][j+1].backwards_nodes.append([(count-1,j),"down"])
            count = count - 1

        return
   
   
   

class options_chain:
    
    def __init__(self, ticker, method, steps):
        self.ticker = ticker
        self.options_chains_dict = {}
        self.method = method
        self.steps = steps
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
    
    def newton_raphson_method_black_scholes(self, epsilon,stock_price, option_price,risk_free_rate, contin_div_yield, time_to_expiration, strike_price):
        sigma = 0.8
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
    
    def dates_to_expiration_fraction(self, expiration_date):
        expiration_date = dt.datetime.strptime(expiration_date, "%Y-%m-%d").date()
        days_to_expiration = (expiration_date - dt.date.today()).days
        fraction_of_days = days_to_expiration/365
        return fraction_of_days
    
    def dates_to_expiration_days(self,expiration_date):
        expiration_date = dt.datetime.strptime(expiration_date, "%Y-%m-%d").date()
        days_to_expiration = (expiration_date - dt.date.today()).days
        return days_to_expiration
    
    def num_dates_to_expir(self,expiration_date):
        expiration_date = dt.datetime.strptime(expiration_date, "%Y-%m-%d").date()
        days_to_expiration = (expiration_date - dt.date.today()).days

        return days_to_expiration
    
    def calc_implied_volatility(self,steps):
        if self.method == 'Black Scholes':
            for key in self.options_chains_dict.keys():
                date_fraction = self.dates_to_expiration_fraction(key)
                self.options_chains_dict[key]['calcImpliedVol'] = np.vectorize(self.newton_raphson_method_black_scholes)(1e-5, self.last_stock_price,\
                                                                                                                         self.options_chains_dict[key]['midpoint'],\
                                                                                                                            self.risk_free,\
                                                                                                                                self.dividend_yield,\
                                                                                                                                    date_fraction,\
                                                                                                                                        self.options_chains_dict[key]['strike'])
                self.options_chains_dict[key]['daystoExpir'] = np.vectorize(self.num_dates_to_expir)(key)

        if self.method == "BinTree Continuous Deriv":
                count = 0
                for key in self.options_chains_dict.keys():
                    dates_to_expiration = self.dates_to_expiration_days(key)
                    self.options_chains_dict[key]['daystoExpir'] = np.vectorize(self.num_dates_to_expir)(key)
                    self.options_chains_dict[key]['calcImpliedVol'] = np.vectorize(self.fsolve_wrapper_binom)(0.6,\
                                                                                                            self.options_chains_dict[key]['strike'],30,\
                                                                                                                self.last_stock_price,\
                                                                                                                    self.risk_free,dates_to_expiration,\
                                                                                                                        self.dividend_yield,\
                                                                                                                            self.options_chains_dict[key]['midpoint'])
                    count = count +1
                    if count > steps:
                        break
        return
    
    def fsolve_wrapper_binom(self, init_imp_vol_guess, strk_pr,number_of_layers, init_stock_pr,intrs_rat,tim_to_expr,stock_div,last_price):
        args = (strk_pr,number_of_layers,init_stock_pr,intrs_rat,tim_to_expr,stock_div,last_price)
        calc_imp_vol = fsolve(self.binomial_tree_objective_function, x0 = init_imp_vol_guess, args=args)
        print(calc_imp_vol)

        return calc_imp_vol
    
    def binomial_tree_objective_function(self, implied_vol, strike_price,number_of_layers, initial_stock_price, interest_rate,time_to_expiration,stock_div,last_price):
        bin_tree = binomial_tree(strike_price=strike_price,number_of_layers=number_of_layers,\
                                 initial_stock_price=initial_stock_price,initial_sigma=implied_vol,\
                                    interest_rate=interest_rate,time_to_expiration=time_to_expiration,stock_dividend=stock_div)
        
        return bin_tree.final_option_value - last_price


    def fetch_options_data(self, ticker):
        self.ticker = yf.Ticker(ticker)
        expiration_dates = self.ticker.options
        self.info = self.ticker.info
        exp = expiration_dates[2]
        self.options_chains_dict = {exp : self.ticker.option_chain(exp).calls for exp in expiration_dates}
        for exp, df in self.options_chains_dict.items():
            if 'bid' in df.columns and 'ask' in df.columns:
                df['midpoint'] = (df['bid'] + df['ask']) / 2
        self.last_stock_price = self.info.get('regularMarketPrice') or ticker.history(period="1d")['Close'][0]
        self.risk_free = yf.Ticker("^IRX").history(period="1d")['Close'][-1]/100
        if yf.Ticker(ticker).info.get('dividendYield') != None:
            self.dividend_yield = yf.Ticker(ticker).info.get('dividendYield')/100
        else:
            self.dividend_yield = 0

        self.calc_implied_volatility(self.steps)


    
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

        lower_strike = last_price * 0.8
        upper_strike = last_price * 1.2

        max_days_to_exp = 150

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
    options = options_chain("NVDA", "BinTree Continuous Deriv", 15)
    options_scholes = options_chain("NVDA", "Black Scholes",None)

    options.plot_imp_vol_surface()
    options_scholes.plot_imp_vol_surface()

if __name__ == "__main__":
    main()
