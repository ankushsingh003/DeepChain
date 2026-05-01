import numpy as np
import pandas as pd

def strategy_sector_rotation(df_dict, lookback=20):
    """
    Strategy: Sector Rotation (Momentum-based)
    Logic: Rank sectors by N-day return. Go long the top performers.
    """
    prices = pd.DataFrame({k: v['Close'] for k, v in df_dict.items()}).dropna()
    momentum = prices.pct_change(lookback)
    
    # Simple logic: signal for each asset if it's in the top 2
    signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    for i in range(lookback, len(prices)):
        top2 = momentum.iloc[i].nlargest(2).index
        signals.loc[signals.index[i], top2] = 1
    return signals
