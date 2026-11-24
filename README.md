# Plutus Backtest

Three Python engines for evaluating timestamped trading signals against returns or OHLC bars. The engines were extracted from the 2021–2022 MyPlutus research project.

## Run a synthetic strategy

From a checkout with Python 3.11 or newer:

```sh
python -m pip install .
python examples/demo.py
```

The [demo](examples/demo.py) uses five synthetic return observations, shifts signals by one bar, and charges a fixed cost when exposure changes. It prints the applied signal, per-bar P&L, and cumulative P&L. Supply a sorted, unique datetime index and document your own signal timing and fill assumptions.

| Engine | Model |
| --- | --- |
| `VecBacktest` | Applies timestamped signals to a return series |
| `VecWithPosAdjBacktest` | Adds point-value and capital calculations |
| `LoopBacktest` | Simulates positions and stop events over OHLC bars, with fills based on the selected bar price |

## History and license

This [MIT-licensed release](LICENSE) retains 21 related source commits from July 2021 to January 2022. The extraction preserves author names and dates, replaces historical email addresses with non-deliverable aliases, and has new commit IDs.
