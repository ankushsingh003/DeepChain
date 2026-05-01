import numpy as np
import pandas as pd

def strategy_scalping_vwap(df):
    """
    Strategy: Scalping (VWAP + EMA micro-structure)
    Logic: Buy when price crosses above VWAP and EMA(9) > EMA(21).
    """
    df = df.copy()
    # VWAP
    df["TP"]      = (df["High"] + df["Low"] + df["Close"]) / 3
    df["Cum_TPV"] = (df["TP"] * df["Volume"]).cumsum()
    df["Cum_Vol"] = df["Volume"].cumsum()
    df["VWAP"]    = df["Cum_TPV"] / df["Cum_Vol"]

    df["EMA9"]  = df["Close"].ewm(span=9).mean()
    df["EMA21"] = df["Close"].ewm(span=21).mean()

    df["signals"] = np.where(
        (df["Close"] > df["VWAP"]) & (df["EMA9"] > df["EMA21"]), 1, 0
    )
    return df
