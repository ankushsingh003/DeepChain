import numpy as np
import pandas as pd

def strategy_mean_reversion_advanced(df, rsi_period=14, bb_period=20, bb_std=2.0):
    """
    Strategy: Mean Reversion (RSI + Bollinger Bands)
    Logic: Buy when RSI < 30 AND price touches lower BB. Sell when RSI > 70 OR price touches upper BB.
    """
    df = df.copy()
    # RSI
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(rsi_period).mean()
    loss  = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs    = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df["BB_Mid"]   = df["Close"].rolling(bb_period).mean()
    df["BB_Std"]   = df["Close"].rolling(bb_period).std()
    df["BB_Upper"] = df["BB_Mid"] + bb_std * df["BB_Std"]
    df["BB_Lower"] = df["BB_Mid"] - bb_std * df["BB_Std"]

    # Signals
    df["signals"] = 0
    position = 0
    for i in range(bb_period, len(df)):
        rsi   = df["RSI"].iloc[i]
        close = df["Close"].iloc[i]
        upper = df["BB_Upper"].iloc[i]
        lower = df["BB_Lower"].iloc[i]
        if position == 0 and rsi < 30 and close <= lower:
            position = 1
        elif position == 1 and (rsi > 70 or close >= upper):
            position = 0
        df.iloc[i, df.columns.get_loc("signals")] = position
    return df
