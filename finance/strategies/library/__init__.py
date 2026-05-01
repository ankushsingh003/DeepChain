from .trend_following import strategy_trend_following
from .swing_trading import strategy_swing_trading
from .mean_reversion_advanced import strategy_mean_reversion_advanced
from .momentum_breakout import strategy_momentum_breakout
from .scalping_vwap import strategy_scalping_vwap
from .pairs_trading import strategy_pairs_trading
from .carry_trade import strategy_carry_trade
from .macd_algo import strategy_macd_algo
from .smart_money_concepts import strategy_smart_money
from .sector_rotation import strategy_sector_rotation

STRATEGIES = {
    "Trend_Following": strategy_trend_following,
    "Swing_Trading": strategy_swing_trading,
    "Advanced_Mean_Reversion": strategy_mean_reversion_advanced,
    "Momentum_Breakout": strategy_momentum_breakout,
    "Scalping_VWAP": strategy_scalping_vwap,
    "Pairs_Trading": strategy_pairs_trading,
    "Carry_Trade": strategy_carry_trade,
    "MACD_Algo": strategy_macd_algo,
    "Smart_Money_Concepts": strategy_smart_money,
    "Sector_Rotation": strategy_sector_rotation
}
