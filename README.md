Calculates the implied volatility curves using the Binomial Asset Pricing model and Black Scholes Approximation via first order derivative approximation method. This implementation of the Binomial Asset Pricing model is fast, written in Python and Numpa, and calculates a 100-layer deep binomial tree in approximately 0.001 seconds per option contract on a laptop CPU, with results consistent with numbers published by financial institutions. Results are best near the money and the Binomial Asset Pricing model utilizes an early exercise option modeled with a continuous dividend yield, to account for the effects of high dividends on early exercise in American Options. The Black Scholes method does not model dividends and is more accurate for European Options and options with 0% dividend yields.


![impliedvolge](https://github.com/user-attachments/assets/78cd6f64-5c3e-4136-aab7-3879f2f2ae4a)

A summary of the Binomial Asset Pricing model can be found in the paper : "Numerical Methods versus Bjerksund and Stensland
Approximations for American Options Pricing"
Marasovic Branka, Aljinovic Zdravka, Poklepovic Tea 
