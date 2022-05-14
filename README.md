# CVaR-Portfolio
Conditional value-at-risk (CVaR) Portfolio Optimization in High Dimensions.







## Scripts

Our algothim is implemented in `opt_algo.py`; `utils.py` contains utils functions.

- Estimation error bound: `estimation_err.py`
- Simulation: `run_simu.py` which requires a sample covariance matrix `Sigma.npy`.
- Real data: `run_real.py`, which requires the S\&P stock data. See the subsection at the end for more details about the data.

The script `summary.py` outputs the results and reproduces the figures.


## Dependencies

This code is delivered via the files described above.

Python (version 3.6 or later) is required to run the files, and it has only been tested on the Linux and the MacOS platforms.


Python packages to run reproducible code:

- cvxopt=1.2.7
- cvxpy=1.2.0
- joblib=1.1.0
- nonlinshrink=0.7
- numba=0.51.1
- numpy=1.21.2
- pandas=1.3.4
- statsmodels=0.13.2
- scikit-learn=1.0.2
- scipy=1.7.1
- tqdm=4.62.3


## S\&P 500 Data
Please refer to [repo](https://github.com/jaydu1/SparsePortfolio/tree/supplement) for obtaining the S\&P 500 dataset.