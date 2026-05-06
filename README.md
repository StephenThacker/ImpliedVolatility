End-to-end database system that scrapes nightly API data and calculates implied volatility curves using Black-Scholes and the Binomial Asset Pricing Model. Uses Docker to manage a containerized PostgreSQL database and a containerized cronjob module (Supercronic) for nightly updates. Data is obtained from a variety of sources, including web scraping, ThetaData API, Yahoo Finance and Polygon.io API. Surfaces can be visualized and animated via a Streamlit user inteface. A summary of the mathematical models used to solve the IV surfaces can be found below. 

<img width="601" height="543" alt="image" src="https://github.com/user-attachments/assets/7c439d22-ff5d-4597-98bc-3dcabdf9bedf" />

A summary of the Binomial Asset Pricing model can be found in the paper : "Numerical Methods versus Bjerksund and Stensland
Approximations for American Options Pricing"
Marasovic Branka, Aljinovic Zdravka, Poklepovic Tea 
