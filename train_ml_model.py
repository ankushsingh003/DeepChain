#!/usr/bin/env python3
"""
train_ml_model.py — DeepChain
Standalone script to train the Strategy ML model.

Run this ONCE before starting the server (or call POST /finance/ml-train via the API).

Usage:
  python train_ml_model.py           # Full training: 30 Nifty-50 symbols (~25 min)
  python train_ml_model.py --quick   # Quick training: 10 symbols (~5 min)

What it does:
  1. Downloads 2 years of OHLCV data for each training symbol from yfinance
  2. Runs all 10 strategies from finance/strategies/library/ against each stock
  3. Computes 15 technical features per stock
  4. Labels each sample: which strategy got the best Sharpe ratio
  5. Trains RandomForest classifier (strategy selector) + regressor (Sharpe predictor)
  6. Saves 4 model files to /models/:
       strategy_classifier.joblib  — maps features → best strategy
       sharpe_regressor.joblib     — maps features → expected Sharpe
       feature_scaler.joblib       — StandardScaler fit on training features
       label_map.joblib            — strategy name ↔ class index mapping
       training_log.json           — training summary + metadata

After training, the /finance/ml-stock-advisory API endpoint uses these models
to advise on any stock in real time.
"""

import sys
import json
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training_run.log"),
    ]
)

def main():
    parser = argparse.ArgumentParser(description="Train DeepChain Strategy ML Model")
    parser.add_argument("--quick",   action="store_true",
                        help="Quick mode: train on 10 symbols instead of 30 (~5 min vs ~25 min)")
    parser.add_argument("--symbols", nargs="*",
                        help="Optional custom list of symbols, e.g. RELIANCE.NS TCS.NS INFY.NS")
    args = parser.parse_args()

    from finance.ml_engine.trainer import StrategyMLTrainer

    trainer = StrategyMLTrainer()

    symbols = args.symbols if args.symbols else None

    print("\n" + "="*60)
    print("  DeepChain Strategy ML Trainer")
    print("="*60)
    if args.quick:
        print("  Mode: QUICK (10 Nifty-50 symbols)")
    elif symbols:
        print(f"  Mode: CUSTOM ({len(symbols)} symbols)")
    else:
        print("  Mode: FULL (30 Nifty-50 symbols)")
    print("="*60 + "\n")

    result = trainer.train(symbols=symbols, quick=args.quick)

    print("\n" + "="*60)
    print("  Training Complete!")
    print("="*60)
    print(json.dumps({
        "samples_trained":       result["samples_trained"],
        "strategies_discovered": result["strategies_seen"],
        "strategy_distribution": result["strategy_distribution"],
        "top_5_features":        result["top_features"],
        "models_saved":          result["models_saved"],
    }, indent=2, default=str))
    print("\n✅ Models saved to /models/")
    print("   Start the server and use POST /finance/ml-stock-advisory to advise on any stock.\n")


if __name__ == "__main__":
    main()
