import pandas as pd
import numpy as np
from empyrical import alpha_beta, sharpe_ratio, max_drawdown, annual_return, tail_ratio


class BasicDesc:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def backtest_eval(self):
        return self.evaluate(self.df)

    @staticmethod
    def evaluate(res_df):
        if 'date' not in res_df.columns.tolist():
            res_df['date'] = res_df.index.date
        pnl_sum = res_df.groupby('date')['pnl'].sum()
        bench_pnl_sum = res_df.groupby('date')['benchmark_pnl'].sum()
        (alpha, beta) = alpha_beta(pnl_sum, bench_pnl_sum, period='daily')
        sharpe = sharpe_ratio(pnl_sum, period='daily')
        bench_sharpe = sharpe_ratio(bench_pnl_sum, period='daily')
        max_down = max_drawdown(pnl_sum)
        wnl_ratio = len(pnl_sum[pnl_sum > 0]) / len(pnl_sum[pnl_sum != 0]) if len(pnl_sum[pnl_sum != 0]) != 0 else 0
        ann_return_strat = annual_return(pnl_sum, period='daily')
        ann_return_bench = annual_return(bench_pnl_sum, period='daily')
        t_r = tail_ratio(pnl_sum)
        sig_sample = res_df['signal'].fillna(0)
        hold_period = np.count_nonzero(sig_sample)
        trade_period = abs(sig_sample - sig_sample.shift().fillna(0)).fillna(0)
        turnover = trade_period.sum() / hold_period if hold_period != 0 else 0

        full_profit = abs(res_df['benchmark_pnl']).sum()
        factor_pnl = res_df['pnl'].sum()
        extraction_ratio = factor_pnl / full_profit

        return {
            'alpha': round(alpha, 2),
            'beta': round(beta, 2),
            'sharpe': round(sharpe, 2),
            'benchmark_sharpe': round(bench_sharpe, 2),
            'max_drawdown': round(max_down, 2),
            'factor_pnl': res_df['pnl'].sum(),
            'factor_annual_pnl': ann_return_strat,
            'benchmark_pnl': res_df['benchmark_pnl'].sum(),
            'benchmark_annual_pnl': ann_return_bench,
            'tail_ratio': round(t_r, 2),
            'turnover': round(turnover, 2),
            'profit_extraction_ratio': round(extraction_ratio, 2),
            'w/l_ratio (exclude flats)': wnl_ratio
        }

    def count_performance(self):
        """count positive/negative returns"""
        perf_df = pd.DataFrame(
            [
                [
                    len(self.df[(self.df['signal'] > 0) & (self.df['pnl'] > 0)]),
                    len(self.df[(self.df['signal'] > 0) & (self.df['pnl'] < 0)]),
                    len(self.df[(self.df['signal'] < 0) & (self.df['pnl'] > 0)]),
                    len(self.df[(self.df['signal'] < 0) & (self.df['pnl'] < 0)]),
                    len(self.df[self.df['signal'] == 0]),
                ],
                [
                    self.df[(self.df['signal'] > 0) & (self.df['pnl'] > 0)]['pnl'].sum(),
                    abs(self.df[(self.df['signal'] > 0) & (self.df['pnl'] < 0)]['pnl'].sum()),
                    self.df[(self.df['signal'] < 0) & (self.df['pnl'] > 0)]['pnl'].sum(),
                    abs(self.df[(self.df['signal'] < 0) & (self.df['pnl'] < 0)]['pnl'].sum()),
                    abs(self.df[self.df['signal'] == 0]['return'].sum())
                ]
            ],
            index=['count', 'total'],
            columns=['long_profit', 'long_loss', 'short_profit', 'short_loss', 'neutral']
        ).T
        return perf_df

    def quantile_describe_pnl(self, valve: float):
        pnl_s = pd.DataFrame(
            [
                [
                    abs(self.df[self.df['pnl'] >= self.df['pnl'].quantile(valve)]['pnl'].sum()),
                    abs(self.df[self.df['pnl'] < self.df['pnl'].quantile(valve)]['pnl'].sum()),
                ],
                [
                    abs(self.df[self.df['pnl'] >= self.df['pnl'].quantile(valve)]['pnl'].mean()),
                    abs(self.df[self.df['pnl'] < self.df['pnl'].quantile(valve)]['pnl'].mean()),
                ]
            ],
            index=['sum', 'avg'],
            columns=['upper', 'lower']
        ).T
        return pnl_s

    @classmethod
    def evaluate_each_signal_performance(
            cls,
            returns: pd.Series,
            signal: pd.DataFrame,
            shift: int,
    ):
        """
        To evaluate all sub signal performance.
        ***
        Notice: the signal df will be loc and clipped according to the index of return series.
                So if you want to evaluate performance under a certain period of time, just pass in the return series
                of that period, and signals will be automatically clipped accordingly.
        :param returns: the return series under which time period that you want to evaluate.
        :param signal: a signal dataframe containing all sub signals
        :param shift: signal shift value
        :return: evaluated result dataframe & all cumulative pnl dataframe.
        """
        signal = signal.loc[signal.index.intersection(returns.index)].reindex(returns.index)
        signal = signal.shift(shift).T.fillna(0)
        eval_df = pd.DataFrame()
        benchmark_cumsum: pd.Series = returns.cumsum()
        benchmark_cumsum.name = 'benchmark'
        pnl_cumsum_df = pd.DataFrame(benchmark_cumsum)
        for k, v in signal.iterrows():
            _pnl: pd.Series = returns * v
            _res_df = pd.DataFrame(
                [_pnl, returns, v], index=['pnl', 'benchmark_pnl', 'signal'], columns=v.index
            ).T
            _eval = pd.DataFrame.from_dict(cls.evaluate(_res_df), orient='index', columns=[k])
            eval_df = pd.concat([eval_df, _eval], axis=1)
            pnl_cumsum_df = pd.concat([pnl_cumsum_df, pd.DataFrame(_pnl.cumsum(), columns=[k])], axis=1)
        return eval_df, pnl_cumsum_df
