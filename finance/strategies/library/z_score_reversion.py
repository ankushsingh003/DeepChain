import pandas as pd
import numpy as np

def z_score_reversion(df: pd.DataFrame, window: int = 20, entry_threshold: float = 2.0) -> pd.DataFrame:
    """
    Strategy: Z-Score Mean Reversion
    Logic: Buy if price is more than 2 std devs below mean. Sell if more than 2 std devs above.
    Formula: Z = (Price - Rolling Mean) / Rolling StdDev
    """
    df = df.copy()
    df['mean'] = df['Close'].rolling(window=window).mean()
    df['std'] = df['Close'].rolling(window=window).std()
    df['z_score'] = (df['Close'] - df['mean']) / df['std']
    
    df['signals'] = 0
    df.loc[df['z_score'] < -entry_threshold, 'signals'] = 1
    df.loc[df['z_score'] > entry_threshold, 'signals'] = -1
    return df
