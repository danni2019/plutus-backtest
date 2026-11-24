"""
backtest logic
another revision.
"""

import pandas as pd
import numpy as np



class VecBacktest:
    """
    A simple basic backtest
    """
    def __init__(
            self,
            pnl: pd.Series
    ):
        self.pnl = pnl

    def plain_backtest(self, sig: pd.Series, sig_shift: int, sig_padding: bool = False, fixed_cost: float = 0):
        res_df = pd.DataFrame(self.pnl.values, index=self.pnl.index, columns=['return'])
        sig = sig.fillna(0)
        sig = sig.copy()
        sig.index = pd.to_datetime(sig.index)
        res_df['signal'] = sig.loc[sig.index.intersection(res_df.index)].reindex(res_df.index)
        if sig_padding:
            res_df['signal'] = res_df['signal'].ffill().shift(sig_shift).fillna(0)
        else:
            res_df['signal'] = res_df['signal'].shift(sig_shift).fillna(0)
        exposure = res_df['signal']
        turnover = exposure.diff().fillna(exposure).abs()
        res_df['pnl'] = res_df['return'] * exposure - fixed_cost * turnover
        res_df['pnl_cumsum'] = res_df['pnl'].cumsum()
        res_df['benchmark_pnl'] = res_df['return']
        res_df['benchmark_pnl_cumsum'] = res_df['benchmark_pnl'].cumsum()
        return res_df
