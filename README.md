This script infers the implied volatility by numerically approximating a solution to the Black-Scholes model for sigma. The script pulls in data from Yahoo Finance's API and uses an analytically derived form of Vega, with the Raphson-Newton root finding method, to solve for the implied volatility.
Please note that this implied volatility is only an approximation, as analytical solutions do not exist for American Styled options and results are best near the money. Moreover, this approximation works best for low/no dividend
yield stocks. Checking a few examples and comparing to publicly recorded at the money implied volatility models, the model performs well under these constraints. Moreover, checking a few stocks with upcoming earnings, 
you can see that the implied volatility reflects this upcoming event. The implied volatility curve was computed on October 17th, 2025.  Outliers, negative volatilities and non-convergent sigmas are filtered.  
![Oct172025](https://github.com/user-attachments/assets/db24167d-f92b-48a4-86c9-9bed41161821)
![Screenshot 2025-10-17 203513](https://github.com/user-attachments/assets/89f31916-5c85-4470-a038-a7c5ebd5b0c9)
