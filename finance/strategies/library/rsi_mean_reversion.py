import pandas as pd
import numpy as np

def rsi_mean_reversion(df: pd.DataFrame, period: int = 14, overbought: int = 70, oversold: int = 30) -> pd.DataFrame:
    """
    Strategy: RSI Mean Reversion
    Logic: Buy when RSI < 30 (oversold), Sell when RSI > 70 (overbought).
    Formula: RSI = 100 - [100 / (1 + (Avg Gain / Avg Loss))]
    """
    df = df.copy()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df['signals'] = 0
    df.loc[df['rsi'] < oversold, 'signals'] = 1
    df.loc[df['rsi'] > overbought, 'signals'] = -1
    return df
