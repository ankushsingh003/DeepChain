from .sma_crossover import sma_crossover
from .rsi_mean_reversion import rsi_mean_reversion
from .bollinger_bands import bollinger_bands_breakout
from .macd_divergence import macd_divergence
from .z_score_reversion import z_score_reversion
from .momentum_roc import momentum_roc
from .parabolic_sar import parabolic_sar_strategy
from .stochastic_oscillator import stochastic_oscillator
from .vwap_strategy import vwap_cross
from .dual_thrust import dual_thrust

STRATEGIES = {
    "SMA_Crossover": sma_crossover,
    "RSI_Mean_Reversion": rsi_mean_reversion,
    "Bollinger_Bands": bollinger_bands_breakout,
    "MACD_Divergence": macd_divergence,
    "Z_Score_Reversion": z_score_reversion,
    "Momentum_ROC": momentum_roc,
    "Parabolic_SAR": parabolic_sar_strategy,
    "Stochastic_Oscillator": stochastic_oscillator,
    "VWAP_Cross": vwap_cross,
    "Dual_Thrust": dual_thrust
}
