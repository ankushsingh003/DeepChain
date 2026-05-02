"""
finance/ml_engine/advisor_engine.py — DeepChain
Module: Stock Strategy Advisor Engine

Full pipeline at inference time:
  1. Fetch live stock data (OHLCV + fundamentals) via StockDataFetcher
  2. Compute the same 15 technical features used during training
  3. Load trained ML model → predict best_strategy + expected_sharpe
  4. Run ALL strategies on 2y history → pick top-3 by Sharpe
  5. If ML prediction agrees with backtest top-1 → use it
     Otherwise → build a HYBRID of ML pick + backtest top-1 with weighted signals
  6. Return full advisory report:
       - recommended strategy name + rationale
       - hybrid flag + blend weights
       - backtest metrics for recommended strategy
       - full stock snapshot (price, PE, fundamentals, technicals)
       - entry/exit guidance
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_DIR       = Path("models")
CLASSIFIER_PATH = MODEL_DIR / "strategy_classifier.joblib"
REGRESSOR_PATH  = MODEL_DIR / "sharpe_regressor.joblib"
SCALER_PATH     = MODEL_DIR / "feature_scaler.joblib"
LABEL_MAP_PATH  = MODEL_DIR / "label_map.joblib"

FEATURE_COLS = [
    "returns_5d", "returns_20d", "returns_60d",
    "rsi", "macd_hist", "bb_pct", "bb_width",
    "vol_20d", "vol_ratio", "vol_spike", "atr_pct",
    "above_sma20", "above_sma50", "above_sma200", "sma20_slope",
]


class StockStrategyAdvisor:
    """
    Inference engine: given a stock symbol, returns the best strategy + full advisory.
    Auto-trains ML model on first run if no saved model found.
    """

    def __init__(self, auto_train_quick: bool = True):
        self.auto_train_quick = auto_train_quick
        self._clf        = None
        self._reg        = None
        self._scaler     = None
        self._label_map  = None
        self._inv_labels: Dict[int, str] = {}
        self._models_loaded = False

    # ── Model Loading / Auto-Training ─────────────────────────────────────────

    def _ensure_models(self):
        """Load models from disk. If not found, auto-train in quick mode."""
        if self._models_loaded:
            return

        if not all(p.exists() for p in [CLASSIFIER_PATH, REGRESSOR_PATH,
                                          SCALER_PATH, LABEL_MAP_PATH]):
            logger.warning("[AdvisorEngine] No trained models found. Auto-training (quick mode)...")
            from .trainer import StrategyMLTrainer
            trainer = StrategyMLTrainer()
            trainer.train(quick=self.auto_train_quick)

        try:
            self._clf       = joblib.load(CLASSIFIER_PATH)
            self._reg       = joblib.load(REGRESSOR_PATH)
            self._scaler    = joblib.load(SCALER_PATH)
            self._label_map = joblib.load(LABEL_MAP_PATH)
            self._inv_labels = {v: k for k, v in self._label_map.items()}
            self._models_loaded = True
            logger.info("[AdvisorEngine] Models loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"[AdvisorEngine] Failed to load models: {e}") from e

    # ── Feature Vector ────────────────────────────────────────────────────────

    def _build_feature_vector(self, df: pd.DataFrame) -> np.ndarray | None:
        """Build the 15-feature vector from price history DataFrame."""
        if len(df) < 60:
            return None
        try:
            close  = df["Close"]
            high   = df["High"]
            low    = df["Low"]
            volume = df["Volume"]
            price  = float(close.iloc[-1])

            returns_5d  = (price / close.iloc[-6]  - 1) * 100 if len(close) > 6  else 0
            returns_20d = (price / close.iloc[-21] - 1) * 100 if len(close) > 21 else 0
            returns_60d = (price / close.iloc[-61] - 1) * 100 if len(close) > 61 else 0

            delta = close.diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rsi   = float((100 - 100 / (1 + gain / loss.replace(0, 1e-9))).iloc[-1])

            ema12    = close.ewm(span=12, adjust=False).mean()
            ema26    = close.ewm(span=26, adjust=False).mean()
            macd     = ema12 - ema26
            sig      = macd.ewm(span=9, adjust=False).mean()
            macd_hist = float((macd - sig).iloc[-1])

            sma20    = close.rolling(20).mean()
            bb_std   = close.rolling(20).std()
            bb_upper = (sma20 + 2 * bb_std).iloc[-1]
            bb_lower = (sma20 - 2 * bb_std).iloc[-1]
            bb_pct   = float((price - bb_lower) / (bb_upper - bb_lower + 1e-9))
            bb_pct   = max(0.0, min(1.0, bb_pct))
            bb_width = float((bb_upper - bb_lower) / (sma20.iloc[-1] + 1e-9))

            log_ret  = np.log(close / close.shift(1))
            vol_20d  = float(log_ret.rolling(20).std().iloc[-1] * np.sqrt(252))
            vol_ratio = float(log_ret.rolling(5).std().iloc[-1] /
                              (log_ret.rolling(20).std().iloc[-1] + 1e-9))

            avg_vol   = float(volume.rolling(20).mean().iloc[-1])
            vol_spike = float(volume.iloc[-1] / (avg_vol + 1))

            tr     = pd.concat([high - low,
                                 (high - close.shift()).abs(),
                                 (low  - close.shift()).abs()], axis=1).max(axis=1)
            atr    = float(tr.rolling(14).mean().iloc[-1])
            atr_pct = atr / price * 100

            sma20_val   = float(sma20.iloc[-1])
            sma50_val   = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else sma20_val
            sma200_val  = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else sma50_val
            sma20_prev  = float(sma20.iloc[-5]) if len(sma20.dropna()) > 5 else sma20_val
            sma20_slope = (sma20_val - sma20_prev) / (sma20_prev + 1e-9) * 100

            vec = [
                returns_5d, returns_20d, returns_60d,
                rsi, macd_hist, bb_pct, bb_width,
                vol_20d, vol_ratio, vol_spike, atr_pct,
                int(price > sma20_val),
                int(price > sma50_val),
                int(price > sma200_val),
                sma20_slope,
            ]
            return np.array([vec], dtype=float)
        except Exception as e:
            logger.warning("[AdvisorEngine] Feature build failed: %s", e)
            return None

    # ── Backtesting helpers ───────────────────────────────────────────────────

    def _backtest_single(self, df: pd.DataFrame, strategy_func) -> Dict[str, float]:
        """Run one strategy → return metrics dict."""
        try:
            data = strategy_func(df.copy())
            if "signals" not in data.columns:
                return {}
            log_ret   = np.log(data["Close"] / data["Close"].shift(1))
            strat_ret = data["signals"].shift(1) * log_ret
            cumret    = np.exp(strat_ret.cumsum())
            total_ret = float(cumret.iloc[-1] - 1)
            std       = float(strat_ret.std())
            sharpe    = float((strat_ret.mean() / std) * np.sqrt(252)) if std > 0 else 0.0
            peak      = cumret.cummax()
            dd        = ((cumret - peak) / peak).min()
            trades    = data[data["signals"].diff().abs() > 0]
            wins      = (strat_ret[data["signals"].shift(1) != 0] > 0).mean()
            return {
                "total_return_pct":  round(total_ret * 100, 2),
                "sharpe_ratio":      round(sharpe, 3),
                "max_drawdown_pct":  round(float(dd) * 100, 2),
                "win_rate_pct":      round(float(wins) * 100, 2) if not np.isnan(wins) else 0.0,
                "trades_count":      len(trades) // 2,
                "final_capital":     round(100000 * (1 + total_ret), 2),
            }
        except Exception as e:
            logger.debug("Backtest error: %s", e)
            return {}

    def _run_all_backtests(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Run all 10 strategies, return list sorted by Sharpe descending."""
        from finance.strategies.library import STRATEGIES
        results = []
        for name, func in STRATEGIES.items():
            metrics = self._backtest_single(df, func)
            if metrics:
                results.append({"strategy": name, **metrics, "func": func})
        results.sort(key=lambda x: x.get("sharpe_ratio", -999), reverse=True)
        return results

    # ── Hybrid Signal Builder ─────────────────────────────────────────────────

    def _build_hybrid(self, df: pd.DataFrame,
                      ml_strategy: str, ml_sharpe: float,
                      bt_strategy: str, bt_sharpe: float,
                      bt_results: List[Dict]) -> Dict[str, Any]:
        """
        When ML pick != backtest top-1, blend their signals:
          weight_ml = ml_sharpe / (ml_sharpe + bt_sharpe)   (Sharpe-weighted)
        Returns combined backtest metrics on blended signals.
        """
        from finance.strategies.library import STRATEGIES

        ml_func = STRATEGIES.get(ml_strategy)
        bt_func = STRATEGIES.get(bt_strategy)
        if ml_func is None or bt_func is None:
            return next((r for r in bt_results if r["strategy"] == bt_strategy), {})

        try:
            df_ml  = ml_func(df.copy())
            df_bt  = bt_func(df.copy())

            w_ml = ml_sharpe / (ml_sharpe + bt_sharpe + 1e-9)
            w_bt = 1.0 - w_ml

            blended_signals = (w_ml * df_ml["signals"] + w_bt * df_bt["signals"])
            # Threshold: > 0.3 → buy (1), < -0.3 → sell (-1), else hold (0)
            final_signals = pd.cut(blended_signals,
                                   bins=[-float("inf"), -0.3, 0.3, float("inf")],
                                   labels=[-1, 0, 1]).astype(float)

            df_hybrid = df.copy()
            df_hybrid["signals"] = final_signals

            log_ret   = np.log(df_hybrid["Close"] / df_hybrid["Close"].shift(1))
            strat_ret = df_hybrid["signals"].shift(1) * log_ret
            cumret    = np.exp(strat_ret.cumsum())
            total_ret = float(cumret.iloc[-1] - 1)
            std       = float(strat_ret.std())
            sharpe    = float((strat_ret.mean() / std) * np.sqrt(252)) if std > 0 else 0.0
            peak      = cumret.cummax()
            dd        = float(((cumret - peak) / peak).min())
            wins      = float((strat_ret[df_hybrid["signals"].shift(1) != 0] > 0).mean())

            return {
                "strategy":         f"Hybrid ({ml_strategy} + {bt_strategy})",
                "is_hybrid":        True,
                "ml_component":     ml_strategy,
                "bt_component":     bt_strategy,
                "ml_weight":        round(w_ml, 3),
                "bt_weight":        round(w_bt, 3),
                "total_return_pct": round(total_ret * 100, 2),
                "sharpe_ratio":     round(sharpe, 3),
                "max_drawdown_pct": round(dd * 100, 2),
                "win_rate_pct":     round(wins * 100, 2) if not np.isnan(wins) else 0.0,
                "final_capital":    round(100000 * (1 + total_ret), 2),
            }
        except Exception as e:
            logger.warning("[AdvisorEngine] Hybrid build failed: %s", e)
            return next((r for r in bt_results if r["strategy"] == bt_strategy), {})

    # ── Entry / Exit Guidance ─────────────────────────────────────────────────

    def _build_entry_exit(self, stock_data: Dict, strategy_name: str,
                          backtest: Dict) -> Dict[str, Any]:
        """Generate concrete entry/exit levels based on technicals."""
        tech   = stock_data.get("technicals", {})
        recent = stock_data.get("recent", {})
        price  = recent.get("current_price", 0)
        atr    = tech.get("atr_14", price * 0.02)
        rsi    = tech.get("rsi_14", 50)

        # Support / resistance
        entry_price  = round(price, 2)
        stop_loss    = round(price - 1.5 * atr, 2)
        target_1     = round(price + 2.0 * atr, 2)
        target_2     = round(price + 3.5 * atr, 2)
        risk_reward  = round((target_1 - entry_price) / (entry_price - stop_loss + 1e-9), 2)

        # Current signal
        if rsi < 35:
            signal_now = "BUY — Oversold conditions"
        elif rsi > 65:
            signal_now = "SELL / AVOID — Overbought conditions"
        elif tech.get("macd_trend") == "Bullish" and tech.get("above_sma20"):
            signal_now = "BUY — Bullish momentum confirmed"
        elif tech.get("macd_trend") == "Bearish" and not tech.get("above_sma20"):
            signal_now = "SELL / WAIT — Bearish momentum"
        else:
            signal_now = "HOLD / WATCH — Neutral conditions"

        return {
            "signal_now":    signal_now,
            "entry_price":   entry_price,
            "stop_loss":     stop_loss,
            "target_1":      target_1,
            "target_2":      target_2,
            "risk_reward":   risk_reward,
            "position_size_note": (
                f"Risk 1-2% of capital per trade. "
                f"With ₹1L capital: max risk ₹{int(100000 * 0.015):,}, "
                f"position size ≈ {int(100000 * 0.015 / max(entry_price - stop_loss, 1))} shares."
            ),
        }

    # ── Main Advisory ──────────────────────────────────────────────────────────

    def advise(self, symbol: str) -> Dict[str, Any]:
        """
        Full advisory pipeline for a stock symbol.
        Returns complete structured advisory dict.
        """
        start = time.time()
        self._ensure_models()

        # 1. Fetch live data
        from .stock_fetcher import StockDataFetcher
        fetcher    = StockDataFetcher()
        stock_data = fetcher.fetch(symbol)
        df         = stock_data["df_history"]

        # 2. Build feature vector
        feat_vec = self._build_feature_vector(df)
        if feat_vec is None:
            raise ValueError(f"Insufficient data for {symbol} to compute features.")

        feat_scaled = self._scaler.transform(feat_vec)

        # 3. ML prediction
        pred_class   = int(self._clf.predict(feat_scaled)[0])
        ml_strategy  = self._inv_labels.get(pred_class, "Trend_Following")
        ml_sharpe    = float(self._reg.predict(feat_scaled)[0])

        # Prediction probabilities for all strategies
        proba = self._clf.predict_proba(feat_scaled)[0]
        ml_confidence = float(proba.max()) * 100
        top3_proba = sorted(
            [(self._inv_labels[i], round(float(p) * 100, 1))
             for i, p in enumerate(proba)],
            key=lambda x: x[1], reverse=True
        )[:3]

        # 4. Run all backtests
        logger.info("[AdvisorEngine] Running backtests for %s ...", symbol)
        bt_results   = self._run_all_backtests(df)
        bt_top       = bt_results[0] if bt_results else {}
        bt_strategy  = bt_top.get("strategy", ml_strategy)
        bt_sharpe    = bt_top.get("sharpe_ratio", 0.0)

        # 5. Decide: ML agrees with backtest? Use it. Otherwise → hybrid.
        is_hybrid    = False
        final_result: Dict[str, Any] = {}

        if ml_strategy == bt_strategy or ml_sharpe <= 0:
            # ML and backtest agree — use backtest-validated strategy
            final_result = bt_top.copy()
            final_result["is_hybrid"] = False
            rationale = (
                f"The ML model (confidence {ml_confidence:.0f}%) and 2-year backtest both "
                f"agree that **{ml_strategy}** is optimal for {stock_data['display_name']}. "
                f"Predicted Sharpe: {ml_sharpe:.2f}. Actual backtested Sharpe: {bt_sharpe:.2f}."
            )
        else:
            # Disagreement → blend for robustness
            is_hybrid    = True
            final_result = self._build_hybrid(df, ml_strategy, max(ml_sharpe, 0.01),
                                               bt_strategy, bt_sharpe, bt_results)
            rationale = (
                f"ML model predicts **{ml_strategy}** (confidence {ml_confidence:.0f}%, "
                f"expected Sharpe {ml_sharpe:.2f}), while 2-year backtest favours "
                f"**{bt_strategy}** (Sharpe {bt_sharpe:.2f}). "
                f"A Sharpe-weighted hybrid is generated for maximum robustness: "
                f"{ml_strategy} at {final_result.get('ml_weight', 0)*100:.0f}% weight + "
                f"{bt_strategy} at {final_result.get('bt_weight', 0)*100:.0f}% weight."
            )

        # 6. Entry / exit guidance
        entry_exit = self._build_entry_exit(stock_data, final_result.get("strategy", ""), final_result)

        # 7. Top-3 backtest strategies for comparison
        top3_bt = [
            {"strategy": r["strategy"],
             "sharpe_ratio": r["sharpe_ratio"],
             "total_return_pct": r["total_return_pct"],
             "max_drawdown_pct": r["max_drawdown_pct"],
             "win_rate_pct": r["win_rate_pct"]}
            for r in bt_results[:3]
        ]

        # Remove non-serialisable func key
        for r in bt_results:
            r.pop("func", None)

        return {
            "symbol":             stock_data["symbol"],
            "display_name":       stock_data["display_name"],
            "sector":             stock_data["sector"],
            "industry":           stock_data["industry"],
            # Recommended strategy
            "recommended_strategy": final_result.get("strategy", ml_strategy),
            "is_hybrid":            final_result.get("is_hybrid", False),
            "rationale":            rationale,
            # ML prediction
            "ml_prediction": {
                "strategy":        ml_strategy,
                "expected_sharpe": round(ml_sharpe, 3),
                "confidence_pct":  round(ml_confidence, 1),
                "top3_probabilities": top3_proba,
            },
            # Backtest performance of recommended strategy
            "backtest_performance": {
                k: v for k, v in final_result.items()
                if k not in ("func", "is_hybrid", "ml_component", "bt_component",
                             "ml_weight", "bt_weight", "strategy")
            },
            # Hybrid details (if applicable)
            "hybrid_details": {
                "ml_component": final_result.get("ml_component"),
                "bt_component": final_result.get("bt_component"),
                "ml_weight":    final_result.get("ml_weight"),
                "bt_weight":    final_result.get("bt_weight"),
            } if is_hybrid else None,
            # Top-3 strategies by backtest Sharpe
            "top3_strategies":   top3_bt,
            # Live stock data
            "fundamentals":      stock_data["fundamentals"],
            "technicals":        stock_data["technicals"],
            "recent":            stock_data["recent"],
            # Entry / exit
            "entry_exit":        entry_exit,
            # Meta
            "latency_sec":       round(time.time() - start, 2),
            "fetch_time":        stock_data["fetch_time"],
        }
