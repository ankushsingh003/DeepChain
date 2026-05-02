"""
finance/ml_engine/stock_fetcher.py — DeepChain
Module: Comprehensive Stock Data Fetcher

Fetches ALL the data needed for a stock advisory:
  - Live OHLCV (yfinance)
  - Fundamentals: PE, PB, EPS, Face Value, Market Cap, Dividend Yield, 52w H/L
  - Technical indicators computed from price history
  - News sentiment placeholder (extensible)
  - NSE/BSE symbol resolution (.NS suffix auto-added for Indian stocks)
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ── Popular Indian stocks auto-mapping ───────────────────────────────────────
INDIAN_SYMBOL_MAP = {
    "RELIANCE":    "RELIANCE.NS",
    "TCS":         "TCS.NS",
    "HDFCBANK":    "HDFCBANK.NS",
    "INFY":        "INFY.NS",
    "INFOSYS":     "INFY.NS",
    "ICICIBANK":   "ICICIBANK.NS",
    "HINDUNILVR":  "HINDUNILVR.NS",
    "ITC":         "ITC.NS",
    "WIPRO":       "WIPRO.NS",
    "SUNPHARMA":   "SUNPHARMA.NS",
    "TATAMOTORS":  "TATAMOTORS.NS",
    "MARUTI":      "MARUTI.NS",
    "ONGC":        "ONGC.NS",
    "NTPC":        "NTPC.NS",
    "SBIN":        "SBIN.NS",
    "AXISBANK":    "AXISBANK.NS",
    "KOTAKBANK":   "KOTAKBANK.NS",
    "BAJFINANCE":  "BAJFINANCE.NS",
    "DRREDDY":     "DRREDDY.NS",
    "CIPLA":       "CIPLA.NS",
    "TATASTEEL":   "TATASTEEL.NS",
    "JSWSTEEL":    "JSWSTEEL.NS",
    "LT":          "LT.NS",
    "BHARTIARTL":  "BHARTIARTL.NS",
    "TITAN":       "TITAN.NS",
    "ADANIENT":    "ADANIENT.NS",
    "ADANIPORTS":  "ADANIPORTS.NS",
    "M&M":         "M&M.NS",
    "HCLTECH":     "HCLTECH.NS",
    "TECHM":       "TECHM.NS",
}


def _resolve_symbol(symbol: str) -> str:
    """Auto-append .NS for Indian stocks if not already suffixed."""
    sym = symbol.upper().strip()
    if sym in INDIAN_SYMBOL_MAP:
        return INDIAN_SYMBOL_MAP[sym]
    if "." not in sym and not sym.endswith("NS") and not sym.endswith("BO"):
        # If it looks like an Indian stock name without suffix, add .NS
        return sym + ".NS"
    return sym


class StockDataFetcher:
    """
    Fetches comprehensive stock data:
      - 2y OHLCV history for ML features + backtesting
      - Live fundamentals (PE, PB, face value, mkt cap, dividend yield, etc.)
      - Derived technical snapshot
    """

    def fetch(self, symbol: str) -> Dict[str, Any]:
        """
        Main entry point. Returns full stock data dict.
        Raises ValueError if symbol not found or data unavailable.
        """
        resolved = _resolve_symbol(symbol)
        logger.info("[StockFetcher] Fetching data for %s (resolved: %s)", symbol, resolved)

        ticker = yf.Ticker(resolved)

        # ── Historical OHLCV ──────────────────────────────────────────────────
        df_2y = ticker.history(period="2y")
        if df_2y.empty:
            raise ValueError(f"No historical data found for '{resolved}'. "
                             f"Check if the symbol is correct (e.g. RELIANCE.NS for NSE).")

        df_6m = ticker.history(period="6mo")

        # ── Fundamentals ──────────────────────────────────────────────────────
        info = {}
        try:
            info = ticker.info or {}
        except Exception as e:
            logger.warning("[StockFetcher] Could not fetch fundamentals for %s: %s", resolved, e)

        fundamentals = self._extract_fundamentals(info, df_2y)

        # ── Technical snapshot ────────────────────────────────────────────────
        technicals = self._compute_technicals(df_2y)

        # ── Recent price action ───────────────────────────────────────────────
        recent = self._recent_price_action(df_6m if not df_6m.empty else df_2y)

        return {
            "symbol":        resolved,
            "display_name":  info.get("longName", resolved),
            "sector":        info.get("sector", "Unknown"),
            "industry":      info.get("industry", "Unknown"),
            "exchange":      info.get("exchange", "Unknown"),
            "currency":      info.get("currency", "INR"),
            "fundamentals":  fundamentals,
            "technicals":    technicals,
            "recent":        recent,
            "df_history":    df_2y,   # full DataFrame for backtesting (not serialised to JSON)
            "fetch_time":    time.time(),
        }

    # ── Fundamentals ─────────────────────────────────────────────────────────

    def _extract_fundamentals(self, info: dict, df: pd.DataFrame) -> Dict[str, Any]:
        def safe(key, default=None):
            v = info.get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return default
            return v

        close = df["Close"].iloc[-1] if not df.empty else 0

        # Face value — yfinance doesn't expose it directly for Indian stocks,
        # so we try a few known fields and fall back to 10 (most common in India)
        face_value = safe("nominalValue") or safe("parValue") or 10.0

        # Market cap in crores for Indian stocks
        mkt_cap_raw = safe("marketCap", 0)
        mkt_cap_cr  = round(mkt_cap_raw / 1e7, 2) if mkt_cap_raw else None

        return {
            "current_price":     round(float(close), 2),
            "pe_ratio":          safe("trailingPE"),
            "pb_ratio":          safe("priceToBook"),
            "eps":               safe("trailingEps"),
            "eps_growth_yoy":    safe("earningsGrowth"),
            "revenue_growth":    safe("revenueGrowth"),
            "face_value":        face_value,
            "book_value":        safe("bookValue"),
            "dividend_yield":    round(safe("dividendYield", 0) * 100, 2),
            "dividend_rate":     safe("dividendRate"),
            "payout_ratio":      safe("payoutRatio"),
            "market_cap":        mkt_cap_raw,
            "market_cap_cr":     mkt_cap_cr,
            "52w_high":          safe("fiftyTwoWeekHigh"),
            "52w_low":           safe("fiftyTwoWeekLow"),
            "50d_avg":           safe("fiftyDayAverage"),
            "200d_avg":          safe("twoHundredDayAverage"),
            "beta":              safe("beta"),
            "debt_to_equity":    safe("debtToEquity"),
            "return_on_equity":  safe("returnOnEquity"),
            "return_on_assets":  safe("returnOnAssets"),
            "profit_margins":    safe("profitMargins"),
            "free_cashflow":     safe("freeCashflow"),
            "analyst_target":    safe("targetMeanPrice"),
            "recommendation":    safe("recommendationKey", "unknown"),
        }

    # ── Technicals ────────────────────────────────────────────────────────────

    def _compute_technicals(self, df: pd.DataFrame) -> Dict[str, Any]:
        close  = df["Close"]
        high   = df["High"]
        low    = df["Low"]
        volume = df["Volume"]

        # RSI
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rsi   = float((100 - 100 / (1 + gain / loss.replace(0, 1e-9))).iloc[-1])

        # MACD
        ema12  = close.ewm(span=12, adjust=False).mean()
        ema26  = close.ewm(span=26, adjust=False).mean()
        macd   = ema12 - ema26
        sig    = macd.ewm(span=9, adjust=False).mean()

        # Bollinger
        sma20     = close.rolling(20).mean()
        bb_std    = close.rolling(20).std()
        bb_upper  = (sma20 + 2 * bb_std).iloc[-1]
        bb_lower  = (sma20 - 2 * bb_std).iloc[-1]
        bb_mid    = sma20.iloc[-1]

        # ATR
        tr     = pd.concat([high - low,
                             (high - close.shift()).abs(),
                             (low  - close.shift()).abs()], axis=1).max(axis=1)
        atr    = float(tr.rolling(14).mean().iloc[-1])

        # Stochastic %K
        low14  = low.rolling(14).min()
        high14 = high.rolling(14).max()
        stoch_k = float((100 * (close - low14) / (high14 - low14 + 1e-9)).iloc[-1])

        price = float(close.iloc[-1])
        vol_avg20 = float(volume.rolling(20).mean().iloc[-1])

        return {
            "rsi_14":           round(rsi, 2),
            "rsi_signal":       "Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral"),
            "macd":             round(float(macd.iloc[-1]), 4),
            "macd_signal":      round(float(sig.iloc[-1]),  4),
            "macd_histogram":   round(float((macd - sig).iloc[-1]), 4),
            "macd_trend":       "Bullish" if macd.iloc[-1] > sig.iloc[-1] else "Bearish",
            "sma_20":           round(float(sma20.iloc[-1]), 2),
            "sma_50":           round(float(close.rolling(50).mean().iloc[-1]), 2),
            "sma_200":          round(float(close.rolling(200).mean().iloc[-1]), 2) if len(close) >= 200 else None,
            "ema_12":           round(float(ema12.iloc[-1]), 2),
            "ema_26":           round(float(ema26.iloc[-1]), 2),
            "bb_upper":         round(float(bb_upper), 2),
            "bb_mid":           round(float(bb_mid),   2),
            "bb_lower":         round(float(bb_lower), 2),
            "bb_position":      round(float((price - bb_lower) / (bb_upper - bb_lower + 1e-9)), 3),
            "atr_14":           round(atr, 2),
            "atr_pct":          round(atr / price * 100, 3),
            "stoch_k":          round(stoch_k, 2),
            "vol_avg_20d":      int(vol_avg20),
            "vol_ratio":        round(float(volume.iloc[-1]) / (vol_avg20 + 1), 3),
            "above_sma20":      price > float(sma20.iloc[-1]),
            "above_sma50":      price > float(close.rolling(50).mean().iloc[-1]),
            "above_sma200":     (price > float(close.rolling(200).mean().iloc[-1])
                                 if len(close) >= 200 else None),
        }

    # ── Recent price action ───────────────────────────────────────────────────

    def _recent_price_action(self, df: pd.DataFrame) -> Dict[str, Any]:
        close = df["Close"]
        if close.empty:
            return {}

        price = float(close.iloc[-1])
        prev  = float(close.iloc[-2]) if len(close) > 1 else price

        return {
            "current_price":   round(price, 2),
            "prev_close":      round(prev, 2),
            "change":          round(price - prev, 2),
            "change_pct":      round((price - prev) / (prev + 1e-9) * 100, 2),
            "day_high":        round(float(df["High"].iloc[-1]), 2),
            "day_low":         round(float(df["Low"].iloc[-1]),  2),
            "volume_today":    int(df["Volume"].iloc[-1]),
            "1w_return_pct":   round((price / close.iloc[-6]  - 1) * 100, 2) if len(close) > 6  else 0,
            "1m_return_pct":   round((price / close.iloc[-22] - 1) * 100, 2) if len(close) > 22 else 0,
            "3m_return_pct":   round((price / close.iloc[-66] - 1) * 100, 2) if len(close) > 66 else 0,
            "6m_return_pct":   round((price / close.iloc[-132] - 1) * 100, 2) if len(close) > 132 else 0,
        }
