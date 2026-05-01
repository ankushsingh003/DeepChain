import pandas as pd
import numpy as np

def vwap_cross(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy: VWAP Cross (Volume Weighted Average Price)
    Logic: Buy when price is below VWAP. Sell when above.
    Formula: VWAP = Sum(Price * Volume) / Sum(Volume)
    """
    df = df.copy()
    # Typical intraday VWAP
    df['tp'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['tpv'] = df['tp'] * df['Volume']
    df['vwap'] = df['tpv'].cumsum() / df['Volume'].cumsum()
    
    df['signals'] = 0
    df.loc[df['Close'] < df['vwap'], 'signals'] = 1
    df.loc[df['Close'] > df['vwap'], 'signals'] = -1
    return df
