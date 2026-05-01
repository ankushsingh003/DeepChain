import pandas as pd
import numpy as np

def bollinger_bands_breakout(df: pd.DataFrame, window: int = 20, num_std: int = 2) -> pd.DataFrame:
    """
    Strategy: Bollinger Bands Breakout
    Logic: Buy when price touches the lower band and starts moving up. Sell when it touches the upper band.
    Formula: Upper = SMA + (k * StdDev), Lower = SMA - (k * StdDev)
    """
    df = df.copy()
    df['sma'] = df['Close'].rolling(window=window).mean()
    df['std'] = df['Close'].rolling(window=window).std()
    df['upper_band'] = df['sma'] + (num_std * df['std'])
    df['lower_band'] = df['sma'] - (num_std * df['std'])
    
    df['signals'] = 0
    df.loc[df['Close'] < df['lower_band'], 'signals'] = 1
    df.loc[df['Close'] > df['upper_band'], 'signals'] = -1
    return df
