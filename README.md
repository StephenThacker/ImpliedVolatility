Calculates the implied volatility curves using the Binomial Asset Pricing model and Black Scholes Approximation via first order derivative approximation method. The Binomial Asset Pricing model is fast, written in Python and Numpa, and calculates a 100-layer deep binomial tree in approximately 0.001 seconds per option contract on a laptop CPU. Results are best near the money and the Binomial Asset Pricing model utilizes an early exercise option modeled with a continuous dividend yield, to account for the effects of high dividends on early exercise in American Options. The Black Scholes method does not model dividends and is more accurate for European Options and options with 0% dividend yields.
![Oct172025](https://github.com/user-attachments/assets/db24167d-f92b-48a4-86c9-9bed41161821)

For the Binomial Asset Pricing Model, a binary tree is created and used to calculate the price of an option. A summary of the method can be found in the paper : "Numerical Methods versus Bjerksund and Stensland
Approximations for American Options Pricing"
Marasovic Branka, Aljinovic Zdravka, Poklepovic Tea 
