"""
backtest logic with position adjustment.
"""

import pandas as pd
import numpy as np
from utils.src.tool.logger import log

logger = log(__file__, 'backtest')


class VecWithPosAdjBacktest:
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
        md['return'] = (md['close'] - md['pre_close']).fillna(0)
        self.df = md

    def plain_backtest(
            self,
            sig: pd.Series,
            max_hands: int,
            market_value_per_point: int,
            leverage: int,
            cushion: float,
    ):
        res_df = self.df.copy()
        sig.index = pd.to_datetime(sig.index)
        res_df['signal'] = sig.loc[sig.index.intersection(res_df.index)].reindex(res_df.index)
        res_df = res_df.loc[res_df[(res_df.index >= sig.index.min()) & (res_df.index <= sig.index.max())].index]
        res_df['signal'] = res_df['signal'].fillna(0)
        res_df['pnl'] = res_df['return'] * res_df['signal']

        # calculate minimum capital required.
        ini_price = res_df['pre_close'].loc[res_df['pre_close'].first_valid_index()]
        min_cap = (max_hands * market_value_per_point * ini_price) / ((leverage + 1) * (1 - cushion))
        logger.info(
            f"Minimum initial capital required: {round(min_cap, 2)} according to initial price: {ini_price} "
            f"with safe cushion: {round(cushion * 100)}% -- Maximum Loss Allowed: {round(min_cap * cushion, 2)}"
        )
        # transform pnl from price diff to percentage based on minimum initial capital
        pnl_cumsum_: pd.Series = (res_df['pnl'] * market_value_per_point).cumsum()
        pnl_cumsum_ = pnl_cumsum_ - [pnl_cumsum_[:i + 1].max() for i in range(len(pnl_cumsum_))]
        pnl_cumsum_ = pd.Series(np.where(pnl_cumsum_ < -(cushion * min_cap), '1', '0'), index=pnl_cumsum_.index).replace(
            to_replace='0', method='pad'
        )

        res_df['pnl'] = (res_df['pnl'] * market_value_per_point) / min_cap
        if '1' in pnl_cumsum_.tolist():
            logger.warning(f"Cushion breached at {pnl_cumsum_[pnl_cumsum_ == '1'].index.min()}")
            res_df['pnl'] = np.where(pnl_cumsum_ == '1', 0, res_df['pnl'])
        res_df['pnl_cumsum'] = res_df['pnl'].cumsum()
        res_df['benchmark_pnl'] = (res_df['return'] * abs(res_df['signal']) * market_value_per_point) / min_cap
        res_df['benchmark_pnl_cumsum'] = res_df['benchmark_pnl'].cumsum()
        return res_df
