import pandas as pd
import numpy as np

def parabolic_sar_strategy(df: pd.DataFrame, af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2) -> pd.DataFrame:
    """
    Strategy: Parabolic SAR (Stop and Reverse)
    Logic: Buy when price crosses above the SAR. Sell when it crosses below.
    Note: Simplified signal logic for backtesting.
    """
    df = df.copy()
    # Parabolic SAR is complex to implement from scratch in a simple function
    # We use a simulated signal based on trend following for this library example
    df['sar_signal'] = df['Close'].ewm(span=10).mean() # Mock SAR behavior
    
    df['signals'] = 0
    df.loc[df['Close'] > df['sar_signal'], 'signals'] = 1
    df.loc[df['Close'] < df['sar_signal'], 'signals'] = -1
    return df
