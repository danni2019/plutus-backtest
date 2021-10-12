"""
backtest using fast for loop. numba accelerated.
"""
import numpy as np
import pandas as pd
from typing import Union
from datetime import time
from numba import jit, float64, int32
from utils.src.tool.logger import log

logger = log(__file__, 'backtest')


@jit(nopython=True)
def _loop(
        action_price: int,
        source_: np.ndarray,
        friction_loss: float = 0
):
    """
    :param action_price:
    :param source_: 'open', 'close', 'high', 'low', 'signal', 'pre_close', 'contract', 'stop_loss' as ndarray
    :param friction_loss:
    :return:
    """
    # record sequences: direction, position, open_price
    trade_record = [float64(0), float64(0), float64(0), int32(1000)]
    pnl, benchmark_pnl = np.array([
        float64(0), float64(0), float64(0), float64(0), 0
    ]), np.array([
        float64(0), float64(0)
    ])
    strat_pnl, bench_price = float64(0), float64(0)

    for v in source_:
        # previous trade record shift.
        prev_record = trade_record
        if bench_price == float64(0):
            bench_price = v[action_price]
        else:
            benchmark_pnl = np.append(benchmark_pnl, [v[action_price] - bench_price, bench_price])
            bench_price = v[action_price]

        # renew current trade record.
        if v[4] != 0:
            _direc = v[4] / abs(v[4])
            _pos = abs(v[4])
        else:
            _direc = float64(0)
            _pos = float64(0)
        trade_record = [
            _direc,
            _pos,
            v[action_price],
            v[6]
        ]
        # whether there is any holding position
        if prev_record[1] == 0:
            # no previous position.
            friction = -trade_record[1] * friction_loss
            prev_p = prev_record[2]
            mark_p = trade_record[2]
            strat_pnl = 0
            stopped = 0
        else:
            if prev_record[3] != trade_record[3]:
                """
                if previously holding position and currently trading a different contract:
                previous position will be moved to current contract
                equivalent of closing out and reopening new position.
                however it is difficult to get the next closing price of an expiring contract
                so right here, we take the position on the previous contract as it was on the current contract
                meaning that we adjust here by replacing the previous open price with the 'pre_close' 
                of the current contract.
                *** Notice here the 'pre_close' label is only reasonable when action_price is set as 'close'
                    and when you are trading at different action_price, 'pre_xxx' will be adjusted automatically.
                    However, you should bear in mind that 'pre_close' is the only label that doesn't contain pricing gap
                """
                if v[7] != 0:
                    if prev_record[0] < 0 and (
                            v[2] - v[5]
                    ) / v[5] > v[7]:
                        strat_pnl = -v[7] * v[5] * prev_record[1]
                        prev_p = v[5]
                        mark_p = v[7] * v[5] + v[5]
                        friction = -(trade_record[1] + prev_record[1]) * friction_loss
                        stopped = 1
                    elif prev_record[0] > 0 and (
                            v[5] - v[3]
                    ) / v[5] > v[7]:
                        strat_pnl = -v[7] * v[5] * prev_record[1]
                        prev_p = v[5]
                        mark_p = v[5] - v[7] * v[5]
                        friction = -(trade_record[1] + prev_record[1]) * friction_loss
                        stopped = 1
                    else:
                        strat_pnl = (trade_record[2] - v[5]) * prev_record[1] * prev_record[0]
                        prev_p = v[5]
                        mark_p = trade_record[2]
                        friction = -abs(trade_record[1] - prev_record[1]) * friction_loss
                        stopped = 0
                else:
                    strat_pnl = (trade_record[2] - v[5]) * prev_record[1] * prev_record[0]
                    prev_p = v[5]
                    mark_p = trade_record[2]
                    friction = -abs(trade_record[1] - prev_record[1]) * friction_loss
                    stopped = 0
            else:
                # if there is any stop loss limit
                if v[7] != 0:
                    if prev_record[0] < 0 and (
                            v[2] - prev_record[2]
                    ) / prev_record[2] > v[7]:
                        strat_pnl = -v[7] * prev_record[2] * prev_record[1]
                        prev_p = prev_record[2]
                        mark_p = v[7] * prev_record[2] + prev_record[2]
                        friction = -(trade_record[1] + prev_record[1]) * friction_loss
                        stopped = 1
                    elif prev_record[0] > 0 and (
                            prev_record[2] - v[3]
                    ) / prev_record[2] > v[7]:
                        strat_pnl = -v[7] * prev_record[2] * prev_record[1]
                        prev_p = prev_record[2]
                        mark_p = prev_record[2] - v[7] * prev_record[2]
                        friction = -(trade_record[1] + prev_record[1]) * friction_loss
                        stopped = 1
                    else:
                        strat_pnl = (trade_record[2] - prev_record[2]) * prev_record[1] * prev_record[0]
                        prev_p = prev_record[2]
                        mark_p = trade_record[2]
                        friction = -abs(trade_record[1] - prev_record[1]) * friction_loss
                        stopped = 0
                else:
                    strat_pnl = (trade_record[2] - prev_record[2]) * prev_record[1] * prev_record[0]
                    prev_p = prev_record[2]
                    mark_p = trade_record[2]
                    friction = -abs(trade_record[1] - prev_record[1]) * friction_loss
                    stopped = 0
        pnl = np.append(pnl, [strat_pnl, prev_p, mark_p, friction, stopped])
    return benchmark_pnl, pnl


class LoopBacktest:
    def __init__(
            self,
            md: pd.DataFrame,
            action_on_0: str,
            action_price: str,
            day_close: Union[str, time] = None
    ):
        """
        A simple backtest logic using for loop.
        :param md: market data of tested instrument
        :param action_on_0: 'close' or 'hold', indicating what actions to be taken on signal 0
        :param action_price: 'open', 'close', or even 'high', 'low' , indicating at which price should trades be done.
        :param day_close: if indicated, positions will be closed out at this specific time.
        """
        md.index = pd.to_datetime(md.index)
        md.sort_index(ascending=True, inplace=True)
        self.md = md
        self.sig = pd.Series(dtype=float)

        self.action_on_0 = action_on_0
        self.action_price = action_price
        if day_close is None:
            self.day_close = False
        else:
            if isinstance(day_close, str):
                self.day_close = time.fromisoformat(day_close)
            else:
                self.day_close = day_close

    def __feed_signal(self, sig: pd.Series):
        self.sig = sig

    def plain_backtest(
            self,
            sig: pd.Series,
            stop_loss: Union[None, pd.Series],
            stop_loss_shift: int = 0,
            friction_loss: float = 0
    ):
        """
        No need to shift signal because this is an imitated event-driven backtest framework,
        all signal will trigger an instant action and return will be calculated at the next event.
        However, a stop_loss_shift is needed for any stop loss data series because generally,
        stop loss is a type of lagging data.
        :param sig:
        :param stop_loss:
        :param stop_loss_shift:
        :param friction_loss:
        :return:
        """
        res_df = self.md.copy()
        sig.index = pd.to_datetime(sig.index)
        res_df['signal'] = sig.loc[sig.index.intersection(res_df.index)].reindex(res_df.index)
        res_df['signal'] = res_df['signal'].fillna(0)

        if stop_loss is not None:
            res_df['stop_loss'] = stop_loss.loc[stop_loss.index.intersection(res_df.index)].reindex(res_df.index)
            res_df['stop_loss'] = abs(res_df['stop_loss']).fillna(0).shift(stop_loss_shift)
        elif 'stop_loss' not in res_df.columns:
            res_df['stop_loss'] = 0
        else:
            res_df['stop_loss'] = abs(res_df['stop_loss'])

        res_df.sort_index(ascending=True, inplace=True)
        if self.day_close:
            res_df['signal'] = np.where(res_df.index.time == self.day_close, 0, res_df['signal'])
        # columns into backtest loop
        passin_cols = ['open', 'close', 'high', 'low', 'signal', f'pre_{self.action_price}', 'contract', 'stop_loss']
        # process 'pre_xxx'
        if f"pre_{self.action_price}" not in res_df.columns:
            res_df[f"pre_{self.action_price}"] = res_df[self.action_price].shift()
        resource_ = res_df[passin_cols].copy()
        # This is important for ensuring types in jit if you cannot utilize string type within mixed ndarray.
        resource_['contract'] = resource_['contract'].apply(lambda x: int(x[-4:]))
        resource_ = resource_.to_numpy()
        benchmark_, strat_ = _loop(
            action_price=passin_cols.index(self.action_price),
            source_=resource_,
            friction_loss=friction_loss,
        )
        strat_ = strat_[5:]
        bench_len = int(len(benchmark_) / 2)
        strat_len = int(len(strat_) / 5)
        benchmark_array = benchmark_.reshape([bench_len, 2])
        strat_array = strat_.reshape([strat_len, 5])
        benchmark_df = pd.DataFrame(
            benchmark_array, columns=['benchmark_pnl', 'bench_open_price'], index=res_df.index
        )
        strat_df = pd.DataFrame(
            strat_array, columns=['pnl', 'open_price', 'mark_price', 'friction_loss', 'is_stopped'], index=res_df.index
        )
        strat_df['pnl'] = strat_df['pnl'] - strat_df['friction_loss']
        strat_df['open_price'] = strat_df['open_price'].replace(to_replace=0, value=np.nan)
        res_df['position_init_price'] = strat_df['open_price']
        res_df['position_markup_price'] = strat_df['mark_price']
        res_df['is_stopped'] = strat_df['is_stopped']
        res_df['pnl_chg'] = strat_df['pnl']
        res_df['pnl_chg_cumsum'] = res_df['pnl_chg'].cumsum()
        res_df['pnl_pctg'] = strat_df['pnl'] / strat_df['open_price']
        res_df['pnl_pctg_cumsum'] = res_df['pnl_pctg'].cumsum()
        res_df['benchmark_pnl_chg'] = benchmark_df['benchmark_pnl']
        res_df['benchmark_pnl_chg_cumsum'] = res_df['benchmark_pnl_chg'].cumsum()
        res_df['benchmark_pnl_pctg'] = benchmark_df['benchmark_pnl'] / benchmark_df['bench_open_price']
        res_df['benchmark_pnl_pctg_cumsum'] = res_df['benchmark_pnl_pctg'].cumsum()
        return res_df


if __name__ == "__main__":

    d = np.array([[1, 2, 3], ])
    f = np.append(d, [1, 1, 1])
    print(f.reshape([2, 3]))