import numpy as np
import pandas as pd

def strategy_pairs_trading(df_a, df_b, window=30, z_entry=2.0, z_exit=0.5):
    """
    Strategy: Statistical Arbitrage (Pairs Trading)
    Logic: Calculate spread z-score. Enter when |z| > 2.0. Exit when |z| < 0.5.
    """
    df = pd.DataFrame(index=df_a.index)
    df['Close_A'] = df_a['Close']
    df['Close_B'] = df_b['Close']
    
    df["Spread"] = df['Close_A'] - (df['Close_A'].rolling(window).corr(df['Close_B']) *
                                   df['Close_A'].rolling(window).std() /
                                   df['Close_B'].rolling(window).std()) * df['Close_B']

    df["Spread_Mean"] = df["Spread"].rolling(window).mean()
    df["Spread_Std"]  = df["Spread"].rolling(window).std()
    df["Z_Score"]     = (df["Spread"] - df["Spread_Mean"]) / df["Spread_Std"]

    df["signals"] = 0
    position = 0
    for i in range(window, len(df)):
        z = df["Z_Score"].iloc[i]
        if position == 0:
            if z > z_entry:   position = -1
            elif z < -z_entry: position = 1
        elif abs(z) < z_exit:
            position = 0
        df.iloc[i, df.columns.get_loc("signals")] = position
    return df
