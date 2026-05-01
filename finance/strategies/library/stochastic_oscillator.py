import pandas as pd
import numpy as np

def stochastic_oscillator(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """
    Strategy: Stochastic Oscillator
    Logic: Buy when %K crosses above 20. Sell when %K crosses below 80.
    Formula: %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
    """
    df = df.copy()
    df['low_min'] = df['Low'].rolling(window=k_period).min()
    df['high_max'] = df['High'].rolling(window=k_period).max()
    df['%K'] = (df['Close'] - df['low_min']) / (df['high_max'] - df['low_min']) * 100
    df['%D'] = df['%K'].rolling(window=d_period).mean()
    
    df['signals'] = 0
    df.loc[df['%K'] < 20, 'signals'] = 1
    df.loc[df['%K'] > 80, 'signals'] = -1
    return df
