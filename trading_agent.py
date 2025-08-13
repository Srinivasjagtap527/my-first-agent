"""
trading_agent.py
==================

This module implements a simple trading agent based on the LE (Level‑and‑EMA) Model
described by options and equity trader **Ellis Dillinger**.  The underlying
idea is to keep trading simple: wait for price to break a key level, then wait
for price to retest either that level or an exponential moving average (EMA)
before entering a trade.  This approach helps filter out false breakouts,
encourages patience and removes the need for dozens of overlapping indicators.

**Disclaimer:** This software is for educational purposes only.  It does not
constitute financial advice and should not be used to execute live trades
without consulting a qualified financial professional.  Trading carries
significant risk and you may lose more than your initial investment.  Use at
your own risk.

Usage example:

.. code-block:: bash

    python trading_agent.py --csv path/to/price_data.csv \
        --level 450.0 --ema 8 --resample 10T

This command loads a CSV file containing historical minute‑level price data,
resamples it to 10‑minute candles, computes the 8‑period EMA and then looks for
breakouts above the ``level`` parameter.  When a breakout and subsequent
retest occur, the script will print a signal indicating a potential long or
short entry.

Input data format
-----------------

The CSV file must contain at least the following columns:

* ``Datetime`` – The timestamp for each bar or trade.  This should be parseable
  by :func:`pandas.to_datetime`.
* ``Open`` – Opening price of the bar.
* ``High`` – High price of the bar.
* ``Low`` – Low price of the bar.
* ``Close`` – Closing price of the bar.

Additional columns such as ``Volume`` are ignored but preserved during
resampling.

Signal logic
------------

1. **Resample** the data into the desired timeframe (default: 10 minutes).
2. **Compute** the EMA on the ``Close`` column using the specified span.
3. **Detect breakouts**:
   - A bullish breakout occurs when the current candle closes above the
     specified ``level`` and the previous candle closed at or below that level.
   - A bearish breakout occurs when the current candle closes below the
     specified ``level`` and the previous candle closed at or above that level.
4. **Wait for a retest** of either the level or the EMA:
   - For bullish setups: look for a subsequent candle whose low dips back to or
     below the level (level retest) or whose low touches the EMA (EMA retest).
     Entry is signalled when the candle closes higher than it opens, showing
     buying strength.
   - For bearish setups: look for a subsequent candle whose high rises back to
     or above the level (level retest) or whose high touches the EMA
     retest).  Entry is signalled when the candle closes lower than it opens.
5. **Report signals** with timestamps and candle details.  The script does not
   place trades or manage risk; it merely identifies potential entry points.

See the ``main`` function at the bottom of this file for CLI usage.
"""

import argparse
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
import numpy as np


@dataclass
class TradeSignal:
    """Represents a potential trade signal.

    Attributes
    ----------
    timestamp: pd.Timestamp
        The time of the signal.
    direction: str
        ``"long"`` for bullish signals (expecting price to rise), ``"short"`` for
        bearish signals (expecting price to fall).
    reason: str
        A brief explanation of why the signal was generated.
    price: float
        The closing price at the time of the signal.
    bar_index: int
        Index of the candle in the resampled DataFrame.
    """

    timestamp: pd.Timestamp
    direction: str
    reason: str
    price: float
    bar_index: int


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Compute the exponential moving average of a price series.

    Parameters
    ----------
    series : pd.Series
        Input price series.
    span : int
        The span for the EMA (commonly 8, 10, 21, etc.).  The smoothing factor
        is calculated as ``alpha = 2 / (span + 1)``.

    Returns
    -------
    pd.Series
        EMA of the input series.
    """
    return series.ewm(span=span, adjust=False).mean()


def resample_to_interval(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV data to a new timeframe.

    Parameters
    ----------
    df : pd.DataFrame
        Original OHLCV data indexed by datetime.
    rule : str
        Resampling rule (e.g. ``'10T'`` for 10‑minute bars).

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame with the same columns.
    """
    agg_map = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
    }
    # Preserve volume if it exists
    if 'Volume' in df.columns:
        agg_map['Volume'] = 'sum'
    return df.resample(rule).agg(agg_map).dropna()


def generate_signals(
    df: pd.DataFrame,
    level: float,
    ema_span: int = 8,
    direction: str = 'both',
) -> List[TradeSignal]:
    """Generate trade signals based on the LE Model.

    Parameters
    ----------
    df : pd.DataFrame
        Resampled OHLCV data with an EMA column.
    level : float
        The key price level to use as the reference for breakouts.  If price
        crosses this level and retests, a signal may be generated.
    ema_span : int, default 8
        Span for the EMA calculation.  Should match the EMA column in `df`.
    direction : {'both', 'long', 'short'}
        If ``'long'``, only bullish signals are returned; if ``'short'``, only
        bearish signals; if ``'both'``, signals of both directions are returned.

    Returns
    -------
    List[TradeSignal]
        List of potential trade signals.
    """
    signals: List[TradeSignal] = []
    df = df.copy()
    ema_col = f'EMA_{ema_span}'
    if ema_col not in df.columns:
        df[ema_col] = compute_ema(df['Close'], ema_span)

    # We need to track whether a breakout has occurred and we are awaiting a retest.
    awaiting_retest_bull = False
    breakout_bar_index_bull: Optional[int] = None
    awaiting_retest_bear = False
    breakout_bar_index_bear: Optional[int] = None

    for idx in range(1, len(df)):
        prev_close = df.iloc[idx - 1]['Close']
        curr_close = df.iloc[idx]['Close']
        curr_open = df.iloc[idx]['Open']
        curr_low = df.iloc[idx]['Low']
        curr_high = df.iloc[idx]['High']
        curr_ema = df.iloc[idx][ema_col]
        timestamp = df.index[idx]

        # Detect bullish breakout
        if direction in ('both', 'long'):
            if (prev_close <= level) and (curr_close > level):
                awaiting_retest_bull = True
                breakout_bar_index_bull = idx
                # print(f"Bullish breakout detected at {timestamp}, waiting for retest")
            # Look for retest if in breakout state
            if awaiting_retest_bull and breakout_bar_index_bull is not None:
                # Level retest: price dips back to or below level
                retest_level = curr_low <= level
                # EMA retest: price dips to EMA
                retest_ema = curr_low <= curr_ema
                # Generate long signal when candle closes higher than it opens
                if (retest_level or retest_ema) and (curr_close > curr_open):
                    reason_parts = []
                    if retest_level:
                        reason_parts.append('level retest')
                    if retest_ema:
                        reason_parts.append('EMA retest')
                    reason = ' & '.join(reason_parts)
                    signals.append(
                        TradeSignal(
                            timestamp=timestamp,
                            direction='long',
                            reason=reason,
                            price=curr_close,
                            bar_index=idx,
                        )
                    )
                    awaiting_retest_bull = False
                    breakout_bar_index_bull = None

        # Detect bearish breakout
        if direction in ('both', 'short'):
            if (prev_close >= level) and (curr_close < level):
                awaiting_retest_bear = True
                breakout_bar_index_bear = idx
            if awaiting_retest_bear and breakout_bar_index_bear is not None:
                retest_level = curr_high >= level
                retest_ema = curr_high >= curr_ema
                if (retest_level or retest_ema) and (curr_close < curr_open):
                    reason_parts = []
                    if retest_level:
                        reason_parts.append('level retest')
                    if retest_ema:
                        reason_parts.append('EMA retest')
                    reason = ' & '.join(reason_parts)
                    signals.append(
                        TradeSignal(
                            timestamp=timestamp,
                            direction='short',
                            reason=reason,
                            price=curr_close,
                            bar_index=idx,
                        )
                    )
                    awaiting_retest_bear = False
                    breakout_bar_index_bear = None

    return signals


def load_price_data(csv_path: str) -> pd.DataFrame:
    """Load price data from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing the OHLC data.  Must include a
        ``Datetime`` column and ``Open``, ``High``, ``Low``, ``Close`` columns.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ``Datetime``.
    """
    df = pd.read_csv(csv_path)
    if 'Datetime' not in df.columns:
        raise ValueError("CSV must contain a 'Datetime' column")
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime')
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple trading agent based on the LE Model (Levels & EMAs)")
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV file with OHLCV data')
    parser.add_argument('--level', type=float, required=True, help='Key price level for breakout detection')
    parser.add_argument('--ema', type=int, default=8, help='EMA span for retest (default: 8)')
    parser.add_argument('--resample', type=str, default='10T', help='Resampling interval (e.g. 10T for 10‑minute bars)')
    parser.add_argument('--direction', type=str, choices=['both', 'long', 'short'], default='both', help='Signal direction to consider')
    args = parser.parse_args()

    df_raw = load_price_data(args.csv)
    df_rs = resample_to_interval(df_raw, args.resample)
    ema_col = f'EMA_{args.ema}'
    df_rs[ema_col] = compute_ema(df_rs['Close'], args.ema)

    signals = generate_signals(df_rs, level=args.level, ema_span=args.ema, direction=args.direction)
    if not signals:
        print("No signals detected based on the provided level and parameters.")
    else:
        print("\nGenerated signals:\n------------------")
        for sig in signals:
            print(
                f"{sig.timestamp} | {sig.direction.upper()} | {sig.reason} | price: {sig.price:.2f}"
            )


if __name__ == '__main__':
    main()
