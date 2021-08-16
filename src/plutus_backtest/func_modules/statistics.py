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
            'alpha': round(alpha, 4),
            'beta': round(beta, 4),
            'sharpe': round(sharpe, 4),
            'benchmark_sharpe': round(bench_sharpe, 4),
            'max_single_day_drawdown': round(pnl_sum.min(), 4),
            'max_drawdown': round(max_down, 4),
            'factor_pnl': res_df['pnl'].sum(),
            'factor_annual_pnl': ann_return_strat,
            'benchmark_pnl': res_df['benchmark_pnl'].sum(),
            'benchmark_annual_pnl': ann_return_bench,
            'tail_ratio': round(t_r, 4),
            'turnover': round(turnover, 4),
            'profit_extraction_ratio': round(extraction_ratio, 4),
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
            signal_df: pd.DataFrame,
            shift: int,
            compounded_signal: pd.Series = None,
            compounded_signal_shift: int = None
    ):
        """
        To evaluate all sub signal performance.
        ***
        Notice: the signal df will be loc and clipped according to the index of return series.
                So if you want to evaluate performance under a certain period of time, just pass in the return series
                of that period, and signals will be automatically clipped accordingly.
        :param returns: the return series under which time period that you want to evaluate.
        :param signal_df: a signal dataframe containing all sub signals
        :param shift: signal shift value
        :param compounded_signal
        :param compounded_signal_shift
        :return: evaluated result dataframe & all cumulative pnl dataframe.
        """
        signal_df = signal_df.loc[signal_df.index.intersection(returns.index)].reindex(returns.index)
        compounded_signal = compounded_signal.loc[compounded_signal.index.intersection(returns.index)].reindex(returns.index)
        signal_df = signal_df.shift(shift).T.fillna(0)
        eval_df = pd.DataFrame()
        benchmark_cumsum: pd.Series = returns.cumsum()
        benchmark_cumsum.name = 'benchmark'
        pnl_cumsum_df = pd.DataFrame(benchmark_cumsum)
        for k, v in signal_df.iterrows():
            _pnl: pd.Series = returns * v
            _res_df = pd.concat(
                [_pnl, returns, v], axis=1
            )
            _res_df.columns = ['pnl', 'benchmark_pnl', 'signal']
            _eval = pd.DataFrame.from_dict(cls.evaluate(_res_df), orient='index', columns=[k])
            eval_df = pd.concat([eval_df, _eval], axis=1)
            pnl_cumsum_df = pd.concat([pnl_cumsum_df, pd.DataFrame(_pnl.cumsum(), columns=[k])], axis=1)
        if compounded_signal is not None and compounded_signal_shift is not None:
            v_com = compounded_signal.shift(compounded_signal_shift)
            _pnl_com = returns * v_com
            _res_df_com = pd.concat(
                [_pnl_com, returns, v_com], axis=1
            )
            _res_df_com.columns = ['pnl', 'benchmark_pnl', 'signal']
            _eval_com = pd.DataFrame.from_dict(cls.evaluate(_res_df_com), orient='index', columns=['compounded'])
            eval_df = pd.concat([eval_df, _eval_com], axis=1)
            pnl_cumsum_df = pd.concat([pnl_cumsum_df, pd.DataFrame(_pnl_com.cumsum(), columns=['compounded'])], axis=1)
        return eval_df, pnl_cumsum_df
