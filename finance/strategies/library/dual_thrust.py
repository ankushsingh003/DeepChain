import pandas as pd
import numpy as np

def dual_thrust(df: pd.DataFrame, n: int = 4, k1: float = 0.5, k2: float = 0.5) -> pd.DataFrame:
    """
    Strategy: Dual Thrust (Range Breakout)
    Logic: Buy when price crosses Open + k1*Range. Sell when Open - k2*Range.
    Formula: Range = Max(HH-LC, HC-LL) over N periods.
    """
    df = df.copy()
    hh = df['High'].rolling(n).max()
    lc = df['Close'].rolling(n).min()
    hc = df['Close'].rolling(n).max()
    ll = df['Low'].rolling(n).min()
    
    df['range'] = np.maximum(hh - lc, hc - ll)
    df['buy_line'] = df['Open'] + (k1 * df['range'])
    df['sell_line'] = df['Open'] - (k2 * df['range'])
    
    df['signals'] = 0
    df.loc[df['Close'] > df['buy_line'], 'signals'] = 1
    df.loc[df['Close'] < df['sell_line'], 'signals'] = -1
    return df
