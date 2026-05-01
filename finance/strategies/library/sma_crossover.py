import pandas as pd
import numpy as np

def sma_crossover(df: pd.DataFrame, short_window: int = 20, long_window: int = 50) -> pd.DataFrame:
    """
    Strategy: SMA Crossover (Trend Following)
    Logic: Buy when Short SMA crosses above Long SMA. Sell when it crosses below.
    Formula: SMA = (Sum of Close Prices over N periods) / N
    """
    df = df.copy()
    df['sma_short'] = df['Close'].rolling(window=short_window).mean()
    df['sma_long'] = df['Close'].rolling(window=long_window).mean()
    
    df['signals'] = 0
    df.loc[df['sma_short'] > df['sma_long'], 'signals'] = 1
    df.loc[df['sma_short'] < df['sma_long'], 'signals'] = -1
    return df
