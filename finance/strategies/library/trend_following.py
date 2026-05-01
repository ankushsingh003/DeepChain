import numpy as np
import pandas as pd

def strategy_trend_following(df, fast=50, slow=200):
    """
    Strategy: Trend Following (Golden Cross / Death Cross)
    Logic: Buy when fast MA crosses above slow MA. Sell when fast MA crosses below slow MA.
    """
    df = df.copy()
    df["MA_Fast"] = df["Close"].rolling(fast).mean()
    df["MA_Slow"] = df["Close"].rolling(slow).mean()

    # Signal: 1 = long, 0 = flat
    df["signals"] = np.where(df["MA_Fast"] > df["MA_Slow"], 1, 0)
    return df
