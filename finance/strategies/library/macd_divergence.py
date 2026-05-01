import pandas as pd
import numpy as np

def macd_divergence(df: pd.DataFrame, fast_span: int = 12, slow_span: int = 26, signal_span: int = 9) -> pd.DataFrame:
    """
    Strategy: MACD Divergence
    Logic: Buy when MACD line crosses above the Signal line. Sell when it crosses below.
    Formula: MACD = EMA(12) - EMA(26), Signal = EMA(MACD, 9)
    """
    df = df.copy()
    df['ema_fast'] = df['Close'].ewm(span=fast_span, adjust=False).mean()
    df['ema_slow'] = df['Close'].ewm(span=slow_span, adjust=False).mean()
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['signal_line'] = df['macd'].ewm(span=signal_span, adjust=False).mean()
    
    df['signals'] = 0
    df.loc[df['macd'] > df['signal_line'], 'signals'] = 1
    df.loc[df['macd'] < df['signal_line'], 'signals'] = -1
    return df
