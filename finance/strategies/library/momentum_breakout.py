import numpy as np
import pandas as pd

def strategy_momentum_breakout(df, lookback=20, vol_mult=1.5):
    """
    Strategy: Momentum / Breakout (Volume-confirmed)
    Logic: Enter long when price breaks 20-day high with volume > 1.5x avg volume.
    """
    df = df.copy()
    df["High_Break"] = df["High"].rolling(lookback).max().shift(1)
    df["Low_Exit"]   = df["Low"].rolling(lookback // 2).min().shift(1)
    df["Avg_Vol"]    = df["Volume"].rolling(lookback).mean().shift(1)
    df["Vol_Spike"]  = df["Volume"] > vol_mult * df["Avg_Vol"]

    df["signals"] = 0
    position = 0
    for i in range(lookback, len(df)):
        close     = df["Close"].iloc[i]
        high_brk  = df["High_Break"].iloc[i]
        low_exit  = df["Low_Exit"].iloc[i]
        vol_ok    = df["Vol_Spike"].iloc[i]
        if position == 0 and close > high_brk and vol_ok:
            position = 1
        elif position == 1 and close < low_exit:
            position = 0
        df.iloc[i, df.columns.get_loc("signals")] = position
    return df
