"""
finance/ml_engine/trainer.py — DeepChain
Module: Strategy ML Trainer

Trains a multi-output ML model that:
  1. Runs ALL 10 strategies from the strategy library on 2 years of historical data
     for a large set of training symbols (Nifty 50 + S&P 500 samples).
  2. Extracts rich technical features from each stock's OHLCV data.
  3. Labels each sample with:
       - which strategy produced the best Sharpe ratio
       - the actual best Sharpe achieved
  4. Trains a RandomForest classifier (best_strategy) + regressor (expected_sharpe).
  5. Saves both models + feature scaler to /models/.

The trained model is then used by StockStrategyAdvisor at inference time:
  Given live features of any stock → predict best strategy + expected Sharpe → run that
  strategy's backtest → return the full analysis.

Usage:
  python -m finance.ml_engine.trainer          # train and save
  python -m finance.ml_engine.trainer --quick  # quick mode (fewer symbols, faster)
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR        = Path("models")
CLASSIFIER_PATH  = MODEL_DIR / "strategy_classifier.joblib"
REGRESSOR_PATH   = MODEL_DIR / "sharpe_regressor.joblib"
SCALER_PATH      = MODEL_DIR / "feature_scaler.joblib"
LABEL_MAP_PATH   = MODEL_DIR / "label_map.joblib"
TRAINING_LOG     = MODEL_DIR / "training_log.json"

# ── Training symbols ──────────────────────────────────────────────────────────
NIFTY_50_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "KOTAKBANK.NS", "AXISBANK.NS", "BAJFINANCE.NS",
    "WIPRO.NS", "SUNPHARMA.NS", "ONGC.NS", "TATAMOTORS.NS", "MARUTI.NS",
    "NTPC.NS", "TITAN.NS", "LTIM.NS", "M&M.NS", "ADANIENT.NS",
    "SBIN.NS", "DRREDDY.NS", "CIPLA.NS", "TATASTEEL.NS", "JSWSTEEL.NS",
    "BAJAJFINSV.NS", "TECHM.NS", "POWERGRID.NS", "HCLTECH.NS", "BHARTIARTL.NS",
]

QUICK_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "WIPRO.NS", "SUNPHARMA.NS", "TATAMOTORS.NS",
]


class StrategyMLTrainer:
    """
    Trains classification + regression models that map stock market features
    to the best-performing strategy from the DeepChain strategy library.
    """

    def __init__(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        # Import all 10 strategies
        from finance.strategies.library import STRATEGIES
        self.strategies: Dict[str, Any] = STRATEGIES

    # ── Feature Engineering ───────────────────────────────────────────────────

    def compute_features(self, df: pd.DataFrame) -> Dict[str, float] | None:
        """
        Compute 25 technical features from OHLCV data.
        Returns None if insufficient data.
        """
        if len(df) < 200:
            return None

        close  = df["Close"]
        high   = df["High"]
        low    = df["Low"]
        volume = df["Volume"]

        try:
            # ── Trend ──────────────────────────────────────────────────────────
            sma20  = close.rolling(20).mean().iloc[-1]
            sma50  = close.rolling(50).mean().iloc[-1]
            sma200 = close.rolling(200).mean().iloc[-1]
            price  = close.iloc[-1]

            # ── Momentum ──────────────────────────────────────────────────────
            returns_5d  = (price / close.iloc[-6]  - 1) * 100 if len(close) > 6  else 0
            returns_20d = (price / close.iloc[-21] - 1) * 100 if len(close) > 21 else 0
            returns_60d = (price / close.iloc[-61] - 1) * 100 if len(close) > 61 else 0

            # ── RSI (14) ───────────────────────────────────────────────────────
            delta = close.diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rsi   = float((100 - 100 / (1 + gain / loss.replace(0, 1e-9))).iloc[-1])

            # ── MACD ───────────────────────────────────────────────────────────
            ema12   = close.ewm(span=12, adjust=False).mean()
            ema26   = close.ewm(span=26, adjust=False).mean()
            macd    = ema12 - ema26
            signal  = macd.ewm(span=9, adjust=False).mean()
            macd_hist = float((macd - signal).iloc[-1])

            # ── Bollinger Bands ────────────────────────────────────────────────
            bb_mid  = close.rolling(20).mean()
            bb_std  = close.rolling(20).std()
            bb_upper = bb_mid + 2 * bb_std
            bb_lower = bb_mid - 2 * bb_std
            bb_pct   = float(((price - bb_lower.iloc[-1]) /
                               (bb_upper.iloc[-1] - bb_lower.iloc[-1] + 1e-9)).clip(0, 1))
            bb_width  = float(((bb_upper - bb_lower) / bb_mid).iloc[-1])

            # ── Volatility ────────────────────────────────────────────────────
            log_ret    = np.log(close / close.shift(1))
            vol_20d    = float(log_ret.rolling(20).std().iloc[-1] * np.sqrt(252))
            vol_ratio  = float(log_ret.rolling(5).std().iloc[-1] /
                               (log_ret.rolling(20).std().iloc[-1] + 1e-9))

            # ── Volume ────────────────────────────────────────────────────────
            avg_vol   = float(volume.rolling(20).mean().iloc[-1])
            vol_spike = float(volume.iloc[-1] / (avg_vol + 1)) if avg_vol > 0 else 1.0

            # ── ATR (14) ───────────────────────────────────────────────────────
            tr     = pd.concat([high - low,
                                 (high - close.shift()).abs(),
                                 (low  - close.shift()).abs()], axis=1).max(axis=1)
            atr    = float(tr.rolling(14).mean().iloc[-1])
            atr_pct = atr / (price + 1e-9) * 100

            # ── Trend strength ────────────────────────────────────────────────
            above_sma20  = int(price > sma20)
            above_sma50  = int(price > sma50)
            above_sma200 = int(price > sma200)
            sma20_slope  = float((sma20 - close.rolling(20).mean().iloc[-5]) /
                                  (close.rolling(20).mean().iloc[-5] + 1e-9) * 100)

            return {
                "price":          round(float(price), 4),
                "sma20":          round(float(sma20), 4),
                "sma50":          round(float(sma50), 4),
                "sma200":         round(float(sma200), 4),
                "returns_5d":     round(returns_5d, 4),
                "returns_20d":    round(returns_20d, 4),
                "returns_60d":    round(returns_60d, 4),
                "rsi":            round(rsi, 4),
                "macd_hist":      round(macd_hist, 6),
                "bb_pct":         round(bb_pct, 4),
                "bb_width":       round(bb_width, 4),
                "vol_20d":        round(vol_20d, 4),
                "vol_ratio":      round(vol_ratio, 4),
                "vol_spike":      round(vol_spike, 4),
                "atr_pct":        round(atr_pct, 4),
                "above_sma20":    above_sma20,
                "above_sma50":    above_sma50,
                "above_sma200":   above_sma200,
                "sma20_slope":    round(sma20_slope, 4),
            }
        except Exception as e:
            logger.warning("Feature computation failed: %s", e)
            return None

    # ── Backtesting ───────────────────────────────────────────────────────────

    def _backtest_strategy(self, df: pd.DataFrame, strategy_func) -> float:
        """Run a strategy and return Sharpe ratio. Returns -999 on failure."""
        try:
            data = strategy_func(df.copy())
            if "signals" not in data.columns:
                return -999.0
            log_ret    = np.log(data["Close"] / data["Close"].shift(1))
            strat_ret  = data["signals"].shift(1) * log_ret
            std        = strat_ret.std()
            if std == 0 or np.isnan(std):
                return 0.0
            sharpe = (strat_ret.mean() / std) * np.sqrt(252)
            return round(float(sharpe), 4) if not np.isnan(sharpe) else 0.0
        except Exception:
            return -999.0

    # ── Dataset builder ───────────────────────────────────────────────────────

    def build_dataset(self, symbols: List[str]) -> pd.DataFrame:
        """
        For each symbol, fetch 2y of data, compute features, run all strategies,
        label with the best strategy. Returns a DataFrame ready for training.
        """
        rows = []
        total = len(symbols)

        for i, symbol in enumerate(symbols):
            logger.info("[Trainer] Processing %s (%d/%d)...", symbol, i + 1, total)
            try:
                df = yf.Ticker(symbol).history(period="2y")
                if len(df) < 200:
                    logger.warning("[Trainer] Skipping %s — insufficient data (%d rows)", symbol, len(df))
                    continue

                features = self.compute_features(df)
                if features is None:
                    continue

                # Run all strategies and pick the best by Sharpe
                sharpes: Dict[str, float] = {}
                for strat_name, strat_func in self.strategies.items():
                    sharpes[strat_name] = self._backtest_strategy(df, strat_func)

                best_strategy = max(sharpes, key=lambda k: sharpes[k])
                best_sharpe   = sharpes[best_strategy]

                if best_sharpe <= 0:
                    logger.debug("[Trainer] Skipping %s — no strategy was profitable.", symbol)
                    continue

                row = {**features,
                       "best_strategy": best_strategy,
                       "best_sharpe":   best_sharpe,
                       "symbol":        symbol}
                rows.append(row)
                logger.info("[Trainer] %s → best=%s sharpe=%.2f", symbol, best_strategy, best_sharpe)

            except Exception as e:
                logger.warning("[Trainer] Failed on %s: %s", symbol, e)

        if not rows:
            raise ValueError("No training data could be built. Check your internet connection and API keys.")

        df_train = pd.DataFrame(rows)
        logger.info("[Trainer] Dataset built: %d samples, %d strategies represented",
                    len(df_train), df_train["best_strategy"].nunique())
        return df_train

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, symbols: List[str] | None = None, quick: bool = False) -> Dict[str, Any]:
        """
        Full training pipeline. Saves models to /models/.
        Returns training summary dict.
        """
        symbols = symbols or (QUICK_SYMBOLS if quick else NIFTY_50_SYMBOLS)
        logger.info("[Trainer] Starting training on %d symbols...", len(symbols))

        df = self.build_dataset(symbols)

        FEATURE_COLS = [
            "returns_5d", "returns_20d", "returns_60d",
            "rsi", "macd_hist", "bb_pct", "bb_width",
            "vol_20d", "vol_ratio", "vol_spike", "atr_pct",
            "above_sma20", "above_sma50", "above_sma200", "sma20_slope",
        ]

        X = df[FEATURE_COLS].fillna(0).values
        y_cls = df["best_strategy"].values
        y_reg = df["best_sharpe"].values

        # Build label map
        label_map = {name: idx for idx, name in enumerate(sorted(set(y_cls)))}
        y_cls_enc = np.array([label_map[s] for s in y_cls])

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ── Classifier: predict best strategy ─────────────────────────────────
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        # Cross-validate if enough data
        if len(X_scaled) >= 20:
            cv_scores = cross_val_score(clf, X_scaled, y_cls_enc,
                                         cv=min(5, len(X_scaled) // 4),
                                         scoring="accuracy")
            logger.info("[Trainer] CV accuracy: %.3f ± %.3f", cv_scores.mean(), cv_scores.std())
        clf.fit(X_scaled, y_cls_enc)

        # ── Regressor: predict expected Sharpe ────────────────────────────────
        reg = RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
        reg.fit(X_scaled, y_reg)

        # ── Save models ────────────────────────────────────────────────────────
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf,       CLASSIFIER_PATH)
        joblib.dump(reg,       REGRESSOR_PATH)
        joblib.dump(scaler,    SCALER_PATH)
        joblib.dump(label_map, LABEL_MAP_PATH)

        # Strategy importance summary
        inv_label_map = {v: k for k, v in label_map.items()}
        feature_importance = dict(zip(FEATURE_COLS, clf.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]

        strategy_dist = df["best_strategy"].value_counts().to_dict()

        summary = {
            "status":            "success",
            "samples_trained":   len(df),
            "strategies_seen":   list(label_map.keys()),
            "strategy_distribution": strategy_dist,
            "top_features":      top_features,
            "models_saved":      [str(CLASSIFIER_PATH), str(REGRESSOR_PATH),
                                   str(SCALER_PATH), str(LABEL_MAP_PATH)],
        }

        # Save training log
        import json, time
        with open(TRAINING_LOG, "w") as f:
            json.dump({**summary, "trained_at": time.time(),
                       "symbols": symbols}, f, indent=2, default=str)

        logger.info("[Trainer] Training complete! %d samples, %d strategies.",
                    len(df), len(label_map))
        return summary

    # ── Check if models exist ─────────────────────────────────────────────────

    @staticmethod
    def models_exist() -> bool:
        return all(p.exists() for p in [CLASSIFIER_PATH, REGRESSOR_PATH,
                                         SCALER_PATH, LABEL_MAP_PATH])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Train on fewer symbols (faster)")
    args = parser.parse_args()

    trainer = StrategyMLTrainer()
    result  = trainer.train(quick=args.quick)
    import json
    print(json.dumps(result, indent=2, default=str))
