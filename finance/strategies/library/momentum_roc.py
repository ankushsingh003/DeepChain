import pandas as pd
import numpy as np

def momentum_roc(df: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """
    Strategy: Momentum Rate of Change (ROC)
    Logic: Buy if ROC is positive. Sell if negative.
    Formula: ROC = [(Current Close - Close n-periods ago) / Close n-periods ago] * 100
    """
    df = df.copy()
    df['roc'] = ((df['Close'] - df['Close'].shift(window)) / df['Close'].shift(window)) * 100
    
    df['signals'] = 0
    df.loc[df['roc'] > 0, 'signals'] = 1
    df.loc[df['roc'] < 0, 'signals'] = -1
    return df
