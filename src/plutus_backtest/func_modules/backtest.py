"""
backtest logic
another revision.
"""

import pandas as pd
from utils.src.tool.logger import log

logger = log(__file__, 'backtest')


class BasicBacktest:
    """
    A simple basic backtest
    """
    def __init__(
            self,
            md: pd.DataFrame,
    ):
        md.index = pd.to_datetime(md.index)
        self.df = md

    def plain_backtest(self, sig: pd.Series):
        """
        signal series should be shifted before pass in as param.
        :param sig:
        :return:
        """
        res_df = self.df.copy()
        sig.index = pd.to_datetime(sig.index)
        res_df['signal'] = sig.loc[sig.index.intersection(res_df.index)].reindex(res_df.index)
        res_df = res_df.loc[res_df[(res_df.index >= sig.index.min()) & (res_df.index <= sig.index.max())].index]
        res_df['signal'] = res_df['signal'].fillna(0)
        res_df['pnl'] = res_df['return'] * res_df['signal']
        res_df['pnl_cumsum'] = res_df['pnl'].cumsum()
        res_df['benchmark_pnl'] = res_df['return']
        res_df['benchmark_pnl_cumsum'] = res_df['benchmark_pnl'].cumsum()
        return res_df



