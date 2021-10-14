"""
backtest using fast for loop. numba accelerated.
"""
import numpy as np
import pandas as pd
from typing import Union
from datetime import time, datetime
from numba import jit, float64, int32
from utils.src.tool.logger import log

logger = log(__file__, 'backtest')


@jit(nopython=True)
def _stop_loss(
        trade_direction: int,
        position: float,
        current_high: float,
        current_low: float,
        current_price: float,
        last_price: float,
        stop_loss_by_diff: float,
        accu_stop_loss_by_diff: float,
):
    if position == 0:
        return 0, last_price, current_price, 0, stop_loss_by_diff, accu_stop_loss_by_diff
    elif trade_direction < 0 and current_high - last_price > stop_loss_by_diff:
        strat_pnl = -stop_loss_by_diff * position
        return strat_pnl, last_price, stop_loss_by_diff + last_price, 1, np.inf, np.inf
    elif trade_direction > 0 and last_price - current_low > stop_loss_by_diff:
        strat_pnl = -stop_loss_by_diff * position
        return strat_pnl, last_price, last_price - stop_loss_by_diff, 1, np.inf, np.inf
    elif trade_direction < 0 and current_high - last_price > accu_stop_loss_by_diff:
        strat_pnl = -accu_stop_loss_by_diff * position
        return strat_pnl, last_price, accu_stop_loss_by_diff + last_price, 1, np.inf, np.inf
    elif trade_direction > 0 and last_price - current_low > accu_stop_loss_by_diff:
        strat_pnl = -accu_stop_loss_by_diff * position
        return strat_pnl, last_price, last_price - accu_stop_loss_by_diff, 1, np.inf, np.inf
    else:
        strat_pnl = (current_price - last_price) * position * trade_direction
        stop_remains = stop_loss_by_diff + ((current_price - last_price) * trade_direction)
        accu_stop_remains = accu_stop_loss_by_diff + ((current_price - last_price) * trade_direction)
        return strat_pnl, last_price, current_price, 0, stop_remains, accu_stop_remains


@jit(nopython=True)
def _loop(
        action_price: int,
        action_on_0: int,
        source_: np.ndarray,
        friction_loss: float = 0,
        stop_loss_type: int = 0,
):
    """
    :param action_price:
    :param action_on_0:
    :param source_: 'open', 'close', 'high', 'low', 'signal', 'pre_close',
                    'contract', 'stop_loss', 'timestamp', 'day_mark', 'accumulative_stop_loss' as ndarray
    :param friction_loss:
    :param stop_loss_type: 0 / 1  0 -- bar stop loss
                                  1 -- daily stop loss
    :return:
    """
    # record sequences: direction, position, open_price, contract
    trade_record = [float64(0), float64(0), float64(0), int32(1000)]
    pnl, benchmark_pnl = np.array([
        float64(0), float64(0), float64(0), float64(0), 0
    ]), np.array([
        float64(0), float64(0)
    ])
    strat_pnl, bench_price = float64(0), float64(0)

    # initiate stop_loss and accu_stop_loss
    stop_diff = accu_stop_diff = np.inf

    for v in source_:
        # previous trade record shift.
        prev_record = trade_record
        if bench_price == float64(0):
            bench_price = v[action_price]
        else:
            benchmark_pnl = np.append(benchmark_pnl, [v[action_price] - bench_price, bench_price])
            bench_price = v[action_price]

        # renew current trade record.
        if v[4] != 0 and v[4] != 333333:
            # valid signal & not on day close
            _direc = v[4] / abs(v[4])
            _pos = abs(v[4])
        elif v[4] == 333333:
            # on day close
            _direc = float64(0)
            _pos = float64(0)
        else:
            # no signal, not on day close
            if action_on_0 == 0:
                _direc = float64(0)
                _pos = float64(0)
            else:
                _direc = prev_record[0]
                _pos = prev_record[1]
        trade_record = [
            _direc,
            _pos,
            v[action_price],
            v[6]
        ]
        if v[7] != 0:
            if stop_loss_type == 0:
                stop_diff = v[7] * trade_record[2]
            else:
                if v[9] == 1:
                    stop_diff = v[7] * trade_record[2]
                elif stop_diff < 0:
                    trade_record[0] = 0
                    trade_record[1] = 0
                    stop_diff = np.inf
        else:
            stop_diff = np.inf
        if v[10] != 0:
            if trade_record[1] != 0 and trade_record[1] * trade_record[0] != prev_record[1] * prev_record[0]:
                accu_stop_diff = v[10] * trade_record[2]
            elif accu_stop_diff < 0:
                trade_record[0] = 0
                trade_record[1] = 0
                accu_stop_diff = np.inf
        else:
            accu_stop_diff = np.inf
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
                    strat_pnl, prev_p, mark_p, stopped, stop_r, acc_stop_r = _stop_loss(
                        trade_direction=prev_record[0],
                        position=prev_record[1],
                        current_high=v[2],
                        current_low=v[3],
                        current_price=trade_record[2],
                        last_price=v[5],
                        stop_loss_by_diff=stop_diff,
                        accu_stop_loss_by_diff=accu_stop_diff
                    )
                    friction = -(trade_record[1] + prev_record[1]) * friction_loss if stopped == 1 else \
                        -abs(trade_record[1] - prev_record[1]) * friction_loss
                    stop_diff = stop_r
                    accu_stop_diff = acc_stop_r
                else:
                    strat_pnl = (trade_record[2] - v[5]) * prev_record[1] * prev_record[0]
                    prev_p = v[5]
                    mark_p = trade_record[2]
                    friction = -abs(trade_record[1] - prev_record[1]) * friction_loss
                    stopped = 0
            else:
                # if there is any stop loss limit
                if v[7] != 0:
                    strat_pnl, prev_p, mark_p, stopped, stop_r, acc_stop_r = _stop_loss(
                        trade_direction=prev_record[0],
                        position=prev_record[1],
                        current_high=v[2],
                        current_low=v[3],
                        current_price=trade_record[2],
                        last_price=prev_record[2],
                        stop_loss_by_diff=stop_diff,
                        accu_stop_loss_by_diff=accu_stop_diff
                    )
                    friction = -(trade_record[1] + prev_record[1]) * friction_loss if stopped == 1 else \
                        -abs(trade_record[1] - prev_record[1]) * friction_loss
                    stop_diff = stop_r
                    accu_stop_diff = acc_stop_r
                else:
                    strat_pnl = (trade_record[2] - prev_record[2]) * prev_record[1] * prev_record[0]
                    prev_p = prev_record[2]
                    mark_p = trade_record[2]
                    friction = -abs(trade_record[1] - prev_record[1]) * friction_loss
                    stopped = 0
        if stopped == 1:
            trade_record[0] = 0
            trade_record[1] = 0
        pnl = np.append(pnl, [strat_pnl, prev_p, mark_p, friction, stopped])
    return benchmark_pnl, pnl


class LoopBacktest:
    def __init__(
            self,
            md: pd.DataFrame,
            action_on_0: str,
            action_price: str,
            day_start: Union[str, time] = None,
            trade_start: Union[str, time] = None,
            day_close: Union[str, time] = None
    ):
        """
        A simple backtest logic using for loop.
        :param md: market data of tested instrument
        :param action_on_0: 'close' or 'hold', indicating what actions to be taken on signal 0
        :param action_price: 'open', 'close', or even 'high', 'low' , indicating at which price should trades be done.
                            'close' price is recommended because this works best with the model.
        :param day_close: if indicated, positions will be closed out at this specific time.
        """
        md.index = pd.to_datetime(md.index)
        md.sort_index(ascending=True, inplace=True)
        self.md = md
        self.sig = pd.Series(dtype=float)

        self.action_on_0 = 0 if action_on_0 == 'close' else 1
        self.action_price = action_price
        if day_start is None:
            self.day_start = False
        else:
            if isinstance(day_start, str):
                self.day_start = time.fromisoformat(day_start)
            else:
                self.day_start = day_start
        if trade_start is None:
            self.trade_start = False
        else:
            if isinstance(trade_start, str):
                self.trade_start = time.fromisoformat(trade_start)
            else:
                self.trade_start = trade_start
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
            stop_loss: Union[None, pd.Series, float],
            stop_loss_period: str,
            accum_stop_loss: Union[None, pd.Series, float],
            friction_loss: float = 0
    ):
        """
        No need to shift signal because this is an imitated event-driven backtest framework,
        all signal will trigger an instant action and return will be calculated at the next event.
        However, a stop_loss_shift is needed for any stop loss data series because generally,
        stop loss is a type of lagging data.
        :param sig:
        :param stop_loss:
        :param stop_loss_period: should be 'bar' or 'day'
                                -- meaning stop loss should be triggered on each bar(for minute/hour level data)
                                 or each day
        :param accum_stop_loss:
        :param friction_loss:
        :return:
        """
        res_df = self.md.copy()
        sig.index = pd.to_datetime(sig.index)
        res_df['signal'] = sig.loc[sig.index.intersection(res_df.index)].reindex(res_df.index)
        res_df['signal'] = res_df['signal'].fillna(0)

        if isinstance(stop_loss, pd.Series):
            res_df['stop_loss'] = stop_loss.loc[stop_loss.index.intersection(res_df.index)].reindex(res_df.index)
            res_df['stop_loss'] = abs(res_df['stop_loss']).fillna(method='pad')
        elif 'stop_loss' not in res_df.columns:
            res_df['stop_loss'] = stop_loss if stop_loss is not None else 0
        else:
            res_df['stop_loss'] = abs(res_df['stop_loss'])

        if isinstance(accum_stop_loss, pd.Series):
            res_df['accum_stop_loss'] = accum_stop_loss.loc[
                accum_stop_loss.index.intersection(res_df.index)
            ].reindex(res_df.index)
            res_df['accum_stop_loss'] = abs(res_df['accum_stop_loss']).fillna(method='pad')
        elif 'accum_stop_loss' not in res_df.columns:
            res_df['accum_stop_loss'] = accum_stop_loss if accum_stop_loss is not None else 0
        else:
            res_df['accum_stop_loss'] = abs(res_df['accum_stop_loss'])

        res_df.sort_index(ascending=True, inplace=True)
        res_df['ts'] = pd.DatetimeIndex(res_df.index).asi8 / (10 ** 9)
        res_df['mk'] = 0
        if self.day_start:
            res_df['mk'] = np.where(res_df.index.time == self.day_start, -1, res_df['mk'])
            res_df = res_df.loc[
                res_df[res_df.index.time >= self.day_start].index
            ]
        if self.trade_start:
            res_df['mk'] = np.where(res_df.index.time == self.trade_start, 1, res_df['mk'])
        else:
            res_df['mk'] = np.where(res_df['mk'] == 0, abs(np.sign(res_df['signal'])), res_df['mk'])
        if self.day_close:
            res_df['signal'] = np.where(res_df.index.time == self.day_close, 333333, res_df['signal'])
            res_df['mk'] = np.where(res_df.index.time == self.day_close, 2, res_df['mk'])
            res_df = res_df.loc[
                res_df[res_df.index.time <= self.day_close].index
            ]

        # columns into backtest loop
        passin_cols = [
            'open', 'close', 'high', 'low', 'signal', f'pre_{self.action_price}', 'contract',
            'stop_loss', 'ts', 'mk', 'accum_stop_loss'
        ]
        # process 'pre_xxx'
        if f"pre_{self.action_price}" not in res_df.columns:
            res_df[f"pre_{self.action_price}"] = res_df[self.action_price].shift()
        resource_ = res_df[passin_cols].copy()
        # This is important for ensuring types in jit if you cannot utilize string type within mixed ndarray.
        resource_['contract'] = resource_['contract'].apply(lambda x: int(x[-4:]))
        resource_ = resource_.to_numpy()

        styp = 0 if stop_loss_period == 'bar' else 1

        benchmark_, strat_ = _loop(
            action_price=passin_cols.index(self.action_price),
            action_on_0=self.action_on_0,
            source_=resource_,
            friction_loss=friction_loss,
            stop_loss_type=styp
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

    a = datetime(2021, 10, 1, 15, 00, 00)
    print(a.date)
    # c = datetime(2021, 10, 2, 15, 00, 00)
    # b = pd.Timestamp(a)
    # # print(b)
    # # print(a.timestamp())
    # # print(c.timestamp())
    # # print(c.timestamp() - a.timestamp())
    # # print(np.inf == np.inf)
    # print(datetime.utcfromtimestamp(1630680900))
    # print(datetime.utcfromtimestamp(1630681200.0))
