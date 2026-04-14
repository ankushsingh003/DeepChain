"""
DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
Module: MLflow Experiment Tracking
"""

import mlflow
import os
from dotenv import load_dotenv

load_dotenv()

class MLflowTracker:
    def __init__(self, experiment_name: str = "GraphRAG_vs_NaiveRAG"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment(experiment_name)

    def log_run(self, params: dict, metrics: dict, run_name: str = None):
        """Logs parameters and metrics to a new MLflow run."""
        with mlflow.start_run(run_name=run_name):
            print(f"[*] Logging run to MLflow: {run_name or 'Untitled'}")
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            print("[+] MLflow logging complete.")

if __name__ == "__main__":
    # Test Logging
    tracker = MLflowTracker()
    tracker.log_run(
        params={"method": "GraphRAG", "top_k": 5},
        metrics={"faithfulness": 0.92, "relevancy": 0.88},
        run_name="Initial_Test_Graph"
    )
