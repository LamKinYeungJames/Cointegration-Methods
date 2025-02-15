import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from pykalman import KalmanFilter


class CointegrationCalculation:
    def __init__(self, stock1: str, stock2: str, rolling_window: int = 252):
        # load in data and compute log prices
        close = pd.read_csv('close_df.csv', index_col=0, parse_dates=True)
        p0 = close.apply(lambda x: next((val for val in x if pd.notnull(val)), None))
        price = np.log(close.div(p0))
        self.stock1 = stock1
        self.stock2 = stock2
        self.data = price[[stock1, stock2]]
        self.data = self.data.dropna(how='any')
        self.tls_res = self.simple_tls(self.data[self.stock1], self.data[self.stock2])
        self.rolling_reg_res = self.rolling_regression(rolling_window)
        self.kf_res = self.kalman_filter_regression()

    def kalman_filter_regression(self, initial_state_window: int = 63) -> pd.DataFrame:
        # use the first initial_state_window to estimate the state mean (not necessary)
        initial_state_data = self.data.iloc[:initial_state_window]
        initial_mean = self.simple_tls(initial_state_data.iloc[:, 0], initial_state_data.iloc[:, 1])[:2]

        # apply kalman filter
        s1, s2 = self.data.iloc[:, 0], self.data.iloc[:, 1]
        kf = KalmanFilter(
            n_dim_state=2,
            n_dim_obs=1,
            initial_state_mean=np.array(initial_mean),
            transition_matrices=np.identity(2),
            transition_offsets=np.zeros(2),
            observation_matrices=sm.add_constant(s1).values[:, np.newaxis],
            observation_offsets=0,
            em_vars=['initial_state_covariance', 'transition_covariance', 'observation_covariance']
        )
        state_mean, state_cov = kf.filter(s2.values)
        kf_beta0, kf_beta1 = state_mean[:, 0], state_mean[:, 1]
        kf_spread = s2.values - kf_beta0 - kf_beta1 * s1.values
        res = pd.DataFrame(zip(kf_beta0, kf_beta1, kf_spread),
                           columns=['kf_beta0', 'kf_beta1', 'kf_spread'],
                           index=self.data.index)
        return res

    def rolling_regression(self, rolling_window: int) -> pd.DataFrame:
        beta0_list = []
        beta1_list = []
        spread_list = []
        for i in range(rolling_window, len(self.data) + 1):
            rolling_data = self.data.iloc[i - rolling_window: i]
            beta0, beta1, spread = self.simple_tls(rolling_data.iloc[:, 0], rolling_data.iloc[:, 1])
            beta0_list.append(beta0)
            beta1_list.append(beta1)
            spread_list.append(spread[-1])
        res = pd.DataFrame(zip(beta0_list, beta1_list, spread_list),
                           columns=['rolling_beta0', 'rolling_beta1', 'rolling_spread'],
                           index=self.data.index[rolling_window - 1:])
        return res

    @staticmethod
    def simple_tls(x: pd.Series, y: pd.Series) -> tuple[float, float, pd.Series]:
        c0 = np.sum((x - x.mean()) * (y - y.mean()))
        c1 = np.sum((x - x.mean()) ** 2 - (y - y.mean()) ** 2)
        c2 = -c0
        beta1 = (-c1 + np.sqrt(c1 ** 2 - 4 * c0 * c2)) / (2 * c0)
        beta0 = y.mean() - beta1 * x.mean()
        residual = y - beta0 - beta1 * x
        return beta0, beta1, residual

    @staticmethod
    def adf_test(spread: pd.Series) -> float:
        adf_res = adfuller(spread)
        t_stat = adf_res[0]
        return t_stat
