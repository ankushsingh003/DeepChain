import numpy as np
import pandas as pd

def strategy_carry_trade(df_high_rate, df_low_rate):
    """
    Strategy: Carry Trade (Interest Rate Differential)
    Logic: Long high-rate currency proxy, Short low-rate currency proxy.
    """
    df = pd.DataFrame(index=df_high_rate.index)
    df["High_Ret"] = df_high_rate["Close"].pct_change()
    df["Low_Ret"] = df_low_rate["Close"].pct_change()

    # Carry signal: always long the spread
    df["Strategy_Return"] = df["High_Ret"] - df["Low_Ret"]
    df["signals"] = 1
    return df
