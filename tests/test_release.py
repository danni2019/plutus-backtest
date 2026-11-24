import pandas as pd
import pytest

from plutus_backtest import LoopBacktest, VecBacktest, VecWithPosAdjBacktest


def test_vectorized_signal_delay_and_cost():
    dates = pd.date_range("2024-04-01", periods=4)
    returns = pd.Series([0.0, 1.0, 2.0, -1.0], index=dates)
    signals = pd.Series([1, -1, -1, 0], index=dates)
    output = VecBacktest(returns).plain_backtest(signals, sig_shift=1, fixed_cost=0.25)
    assert output["signal"].tolist() == [0, 1, -1, -1]
    assert output["pnl"].tolist() == pytest.approx([0, 0.75, -2.5, 1])


def bar_data():
    prices = [100.0, 101.0, 102.0, 101.0]
    index = pd.date_range("2024-04-01 09:00", periods=4, freq="min")
    return pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices, "contract": "DEMO2404"},
        index=index,
    )


def test_position_adjustment_uses_point_value():
    bars = bar_data()
    signals = pd.Series([1, 1, 0, 0], index=bars.index)
    output, flag = VecWithPosAdjBacktest(bars).plain_backtest(
        signals, sig_shift=1, sig_padding=False, max_hands=1, market_value_per_point=10,
        leverage=0, cushion=0.5, show_pnl_by_percentage=False, original_timeframe=True,
    )
    assert output["pnl"].tolist() == pytest.approx([0, 10, 10, 0])
    assert flag == 1
    assert "return" not in bars.columns


def test_event_loop_returns_one_result_per_bar():
    bars = bar_data()
    signals = pd.Series([1, 1, 0, 0], index=bars.index)
    output = LoopBacktest(bars, action_on_0="close", action_price="close").plain_backtest(
        signals, stop_loss=None, stop_profit=None, stop_loss_period="bar",
        stop_profit_period="bar", accum_stop_loss=None,
    )
    assert output.index.equals(bars.index)
    assert output["pnl_chg"].tolist() == pytest.approx([0, 1, 1, 0])
