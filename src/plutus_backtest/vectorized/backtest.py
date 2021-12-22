"""
backtest logic
another revision.
"""

import pandas as pd
import numpy as np
from utils.src.tool.logger import log

logger = log(__file__, 'backtest')


class VecBacktest:
    """
    A simple basic backtest
    """
    def __init__(
            self,
            md: pd.DataFrame,
    ):
        md.index = pd.to_datetime(md.index)
        md.sort_index(ascending=True, inplace=True)
        if "pre_close" not in md.columns:
            md['pre_close'] = md['close'].shift()
        md['return'] = ((md['close'] - md['pre_close']) / md['pre_close']).fillna(0)
        md['to_high'] = ((md['high'] - md['pre_close']) / md['pre_close']).fillna(0)
        md['to_low'] = ((md['low'] - md['pre_close']) / md['pre_close']).fillna(0)
        self.df = md

    def plain_backtest(self, sig: pd.Series, stop_loss: float = 0, signal_padding: bool = False):
        """
        signal series should be shifted before pass in as param.
        :param sig:
        :param stop_loss: decimal stop loss. eg. 0.02 : meaning the stop loss wil be at -2%
        :return:
        """
        if stop_loss < 0:
            raise ValueError("stop_loss should be a float value no less than 0")
        res_df = self.df.copy()
        sig.index = pd.to_datetime(sig.index)
        res_df['signal'] = sig.loc[sig.index.intersection(res_df.index)].reindex(res_df.index)
        res_df = res_df.loc[res_df[(res_df.index >= sig.index.min()) & (res_df.index <= sig.index.max())].index]
        res_df['signal'] = res_df['signal'].fillna(0) if not signal_padding else res_df['signal'].fillna(method='bfill')
        res_df['pnl'] = res_df['return'] * res_df['signal']
        res_df['pnl'] = np.where(
            ((
                    (res_df['signal'] > 0) & (res_df['to_low'] <= -stop_loss)
            ) | (
                    (res_df['signal'] < 0) & (res_df['to_high'] >= stop_loss)
            )) & (stop_loss > 0),
            -stop_loss,
            res_df['pnl']
        )
        res_df['pnl_cumsum'] = res_df['pnl'].cumsum()
        res_df['benchmark_pnl'] = res_df['return']
        res_df['benchmark_pnl_cumsum'] = res_df['benchmark_pnl'].cumsum()
        return res_df



