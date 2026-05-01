# Trading Strategy Knowledge Base (Advanced v2)

This document contains 10 institutional-grade trading strategies for the DeepChain Advisor.

## 1. Trend Following (MA Crossover)
- **Logic**: Buy when fast MA crosses above slow MA.
- **Components**: MA Fast (50), MA Slow (200).

## 2. Swing Trading (Donchian Breakout)
- **Logic**: Enter long on N-day high breakout. Risk-to-reward ratio applied.
- **Components**: Donchian Channels (20d).

## 3. Advanced Mean Reversion (RSI + Bollinger)
- **Logic**: Buy when RSI < 30 and price hits lower Bollinger Band.
- **Components**: RSI (14), BB (20, 2std).

## 4. Momentum Breakout (Volume Confirmed)
- **Logic**: Enter long on N-day high break if volume > 1.5x average.
- **Components**: Volume Multiplier, High Breakout.

## 5. Scalping (VWAP + EMA)
- **Logic**: Intraday scalp. Buy when Price > VWAP and EMA(9) > EMA(21).
- **Components**: VWAP, EMA Cross.

## 6. Statistical Arbitrage (Pairs Trading)
- **Logic**: Trade cointegrated asset pairs based on spread Z-Score.
- **Components**: Hedge Ratio, Z-Score (2.0/0.5).

## 7. Carry Trade
- **Logic**: Profit from interest rate differentials between currencies.
- **Components**: FX Spread.

## 8. Algorithmic MACD
- **Logic**: Buy when MACD crosses above Signal line.
- **Components**: EMA(12/26), Signal(9).

## 9. Smart Money Concepts (SMC)
- **Logic**: Identify Bullish Order Blocks and Break of Structure (BOS).
- **Components**: Swing Highs/Lows, Order Blocks.

## 10. Sector Rotation
- **Logic**: Momentum-based rotation into top-performing market sectors.
- **Components**: N-day momentum ranking.
