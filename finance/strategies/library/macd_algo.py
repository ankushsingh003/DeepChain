import numpy as np
import pandas as pd

def strategy_macd_algo(df, fast=12, slow=26, signal=9):
    """
    Strategy: Algorithmic MACD Crossover
    Logic: Buy when MACD line crosses above Signal line. Sell when it crosses below.
    """
    df = df.copy()
    df["EMA_Fast"]   = df["Close"].ewm(span=fast, adjust=False).mean()
    df["EMA_Slow"]   = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"]       = df["EMA_Fast"] - df["EMA_Slow"]
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()

    df["signals"] = np.where(df["MACD"] > df["MACD_Signal"], 1, 0)
    return df
