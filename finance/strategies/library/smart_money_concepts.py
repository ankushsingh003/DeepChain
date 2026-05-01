import numpy as np
import pandas as pd

def strategy_smart_money(df, lookback=10):
    """
    Strategy: Smart Money Concepts (Order Block + BOS)
    Logic: Identify Bullish Order Blocks and Break of Structure (BOS).
    Enter when price retraces into order block after BOS.
    """
    df = df.copy()
    # Swing highs and lows
    df["Swing_High"] = df["High"].rolling(lookback * 2 + 1, center=True).max()
    df["Is_SH"]      = df["High"] == df["Swing_High"]

    # Break of Structure (BOS)
    df["Prev_SH"]    = df["Swing_High"].shift(lookback)
    df["BOS_Up"]     = df["Close"] > df["Prev_SH"]

    # Order Block: last bearish candle before BOS (approximation)
    df["Bearish_OB"] = (df["Close"] < df["Open"]) & (df["BOS_Up"].shift(-1))

    # Signal: enter when price dips into a recent order block after BOS
    df["OB_High"] = np.where(df["Bearish_OB"], df["High"], np.nan)
    df["OB_Low"]  = np.where(df["Bearish_OB"], df["Low"],  np.nan)
    df["OB_High"] = df["OB_High"].ffill()
    df["OB_Low"]  = df["OB_Low"].ffill()

    df["signals"] = np.where(
        df["BOS_Up"] & (df["Close"] >= df["OB_Low"]) & (df["Close"] <= df["OB_High"]), 1, 0
    )
    return df
