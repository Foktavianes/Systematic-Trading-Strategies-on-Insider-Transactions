import pandas as pd
import numpy as np
import yfinance as yf

from matplotlib import pyplot as plt

class BTResults:
    """Results container with calculation for each ratio
    Assumes a daily returns.
    """
    def __init__(self, total_return_series, benchmark, turnover, deployed_capital):
        self.total_return_series = total_return_series
        self.benchmark = benchmark
        self.turnover = turnover
        self.deployed_capital = deployed_capital

    @staticmethod
    def cal_expected_returns(total_returns):
        """Mean return

        Parameters
        ----------
        total_returns : pd.Series
            Series of total return index

        Returns
        -------
        float
            expected returns
        """

        return total_returns.dropna().pct_change().mean()
    
    @staticmethod
    def cal_returns_deviation(total_returns):
        return total_returns.dropna().pct_change().std()
    
    @classmethod
    def cal_annualized_sharpe(cls, total_returns):
        return cls.cal_expected_returns(total_returns)/cls.cal_returns_deviation(total_returns)
    
    @staticmethod
    def cal_cumulative_drawdown(total_returns):
        return total_returns - total_returns.cummax()
    
    @staticmethod
    def cal_quarterly_drawdown(total_returns):
        return total_returns - total_returns.rolling(25*3).max()
    
    @staticmethod
    def cal_semi_deviation(total_returns):
        return (((total_returns.pct_change()).clip(upper=0) ** 2).sum(0) / total_returns.shape[0]) ** 0.5

    @staticmethod
    def cal_cagr(total_returns):
        """Assuming 252 trading days

        Parameters
        ----------
        total_returns : pd.Series
            a tri series

        Returns
        -------
        float
            CAGR in decimal
        """
        return (total_returns.iloc[-1] ** (252/total_returns.shape[0])) - 1
    
    def expected_period_return(self, bm=False):
        if bm:
            return self.cal_expected_returns(self.benchmark)
        else:
            return self.cal_expected_returns(self.total_return_series)
        
    def period_deviation(self, bm=False):
        if bm:
            return self.cal_returns_deviation(self.benchmark)
        else:
            return self.cal_returns_deviation(self.total_return_series)
        
    def period_semi_deviation(self, bm=False):
        if bm:
            return self.cal_semi_deviation(self.benchmark)
        else:
            return self.cal_semi_deviation(self.total_return_series)
        
    def sharpe(self, bm=False):
        return self.expected_period_return(bm)/self.period_deviation(bm)

    def sortino(self, bm=False):
        return self.expected_period_return(bm)/self.period_semi_deviation(bm)

    def calmer(self, bm=False):
        return -self.cagr(bm)/self.cumulative_drawdown(bm).min()

    def cagr(self, bm=False):
        if bm:
            return self.cal_cagr(self.benchmark)
        else:
            return self.cal_cagr(self.total_return_series)
    
    def cumulative_drawdown(self, bm=False):
        if bm:
            return self.cal_cumulative_drawdown(self.benchmark)
        else:
            return self.cal_cumulative_drawdown(self.total_return_series)
        
    def quick_report(self):
        strat = pd.Series(
            {'Expected Period Return': self.expected_period_return(False),
             'Ann. Sharpe': self.sharpe(False) * (252**0.5),
             'Ann. Sortino': self.sortino(False) * (252**0.5),
             'Calmer': self.calmer(False),
             'Max DD.': self.cumulative_drawdown(False).min(),
             'CAGR': self.cagr(False),
             'Avg. Daily Turnover': self.turnover.mean()},
            name='Strategy'
        )
        bm = pd.Series(
            {'Expected Period Return': self.expected_period_return(True),
             'Ann. Sharpe': self.sharpe(True) * (252**0.5),
             'Ann. Sortino': self.sortino(True) * (252**0.5),
             'Calmer': self.calmer(True),
             'Max DD.': self.cumulative_drawdown(True).min(),
             'CAGR': self.cagr(True)},
            name='Benchmark'
        )

        return pd.concat([strat, bm], axis=1)
    
    def report(self):
        fig, ax = plt.subplots(4, 1, figsize=(12,12), sharex=True, gridspec_kw = {'height_ratios':[8,2,2,2]})

        ax[0].set_title('To tal Returns')
        ax[1].set_title('Relative PnL')
        ax[2].set_title('Drawdown Plot')
        ax[3].set_title('Deployed Capital')


        self.total_return_series.plot(ax=ax[0], label='Strategy TRI')
        self.benchmark.plot(ax=ax[0], label='Benchmark TRI')
        self.cumulative_drawdown(False).plot(ax=ax[2])

        (self.total_return_series - self.benchmark).plot(ax=ax[1], label='Cumulative Alpha')
        (self.total_return_series.pct_change() - self.benchmark.pct_change()).rolling(25).sum().plot(ax=ax[1], label='25 Days rolling diff')
        
        self.deployed_capital.plot(ax=ax[3])

        ax[0].grid()
        ax[1].grid()
        ax[2].grid()
        ax[3].grid()

        ax[0].legend()
        ax[1].legend()

        plt.show()

        return self.quick_report()


class YFCrossSectionalBT:
    def __init__(self, universe, start_date=None, end_date=None):
        try:
            input_type = universe[0]
        except KeyError:
            input_type = universe.iloc[0]

        if isinstance(input_type, str):
            self.yf_data = yf.download([s.replace('.', '-') for s in universe], start=start_date, end=end_date)
            self.adj_open = (self.yf_data['Adj Close']/self.yf_data['Close'])*self.yf_data['Open']
        elif isinstance(input_type, pd.Series):
            self.adj_open = universe.copy()

        # benchmark is sp500
        bm = yf.download('^GSPC', start=start_date, end=end_date).reindex(self.adj_open.index)
        self.bm = (bm['Adj Close']/bm['Close']) * bm['Open']

    def backtest(self, signal, delay, holding_period):
        # Forward return (-1 shift) which is further shifted for each delayed trades day
        shifted_returns = self.adj_open.pct_change().shift(-1-delay)
        # From earliest signal day to latest + holding period
        shifted_returns = shifted_returns.loc[signal.index.min(): signal.index.max() + pd.Timedelta(holding_period, 'd')]

        #weightings are rolling mean of signal/holding period. 
        weightings = (signal.reindex(shifted_returns.index)/holding_period).rolling(holding_period, min_periods=1).sum()

        #returns are then calculated as sum of weights * shifted returns per period
        #cumulative products of this series + 1 results in the total return index
        #we also shift day back to convert the value back to base return, i.e. returns earned on
        #trade closing on the end of holding period.
        cumulative_returns = ((weightings*shifted_returns).sum(1)+1).cumprod().shift(1+delay).dropna()

        # summing the weightings by each rows results in deployed capital per period.
        deployed_capital = weightings.sum(1).reindex(cumulative_returns.index)

        # We use the benchmark multiplied by deployed capital for a fair comparison
        equal_exposure_bm = (deployed_capital*(self.bm.pct_change())+1).cumprod().fillna(1).reindex(cumulative_returns.index)

        # also, turnover of both buy and sell.
        turnover = weightings.diff().abs().sum(1).reindex(cumulative_returns.index)

        return BTResults(cumulative_returns, equal_exposure_bm, turnover, deployed_capital)
    