import pandas as pd

from plutus_backtest import VecBacktest


def main():
    dates = pd.date_range("2024-04-01", periods=5)
    returns = pd.Series([0.0, 0.01, -0.02, 0.03, 0.01], index=dates)
    signals = pd.Series([1, 1, -1, 0, 0], index=dates)
    result = VecBacktest(returns).plain_backtest(signals, sig_shift=1, fixed_cost=0.001)
    print(result[["signal", "pnl", "pnl_cumsum"]].to_string())


if __name__ == "__main__":
    main()
