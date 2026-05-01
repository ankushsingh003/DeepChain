import numpy as np
import pandas as pd

def strategy_swing_trading(df, window=20, rr_ratio=2.0):
    """
    Strategy: Swing Trading (Support / Resistance / Donchian Breakout)
    Logic: Enter long on N-day high breakout. Exit when price falls below N-day low.
    """
    df = df.copy()
    df["High_N"] = df["High"].rolling(window).max().shift(1)
    df["Low_N"]  = df["Low"].rolling(window).min().shift(1)

    df["signals"] = 0
    position = 0
    entry_price = 0

    for i in range(window, len(df)):
        close = df["Close"].iloc[i]
        high_n = df["High_N"].iloc[i]
        low_n  = df["Low_N"].iloc[i]

        if position == 0 and close > high_n:
            position = 1
            entry_price = close
            stop = low_n
            target = entry_price + rr_ratio * (entry_price - stop)
        elif position == 1:
            if close <= low_n or close >= target:
                position = 0
        df.iloc[i, df.columns.get_loc("signals")] = position
    return df
