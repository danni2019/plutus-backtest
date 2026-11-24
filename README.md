# Plutus Backtest

Python backtest engines extracted from MyPlutus. `VecBacktest` applies timestamped signals to a return series. `VecWithPosAdjBacktest` adds point-value and capital calculations. `LoopBacktest` simulates positions and stop events over OHLC bars.

```sh
python -m pip install .
python examples/demo.py
```

The demo uses synthetic input. Vectorized fixed costs are charged on the absolute change in applied exposure. The event loop models fills using the selected bar price. For results that can be reproduced, supply a sorted, unique datetime index and explicit per-bar signals and fill assumptions.

This repository retains 21 relevant source commits from July 2021 to January 2022. The extraction preserves author names and dates, replaces historical email addresses with non-deliverable aliases, and creates new commit IDs. The standalone packaging and compatibility changes are recorded in a separate commit.
