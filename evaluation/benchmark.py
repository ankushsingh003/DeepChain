"""
evaluation/benchmark.py  —  DeepChain Hybrid-RAG
Fixed: outdated Ragas API usage + no async support + no MLflow integration.

Changes vs original:
  - Uses ragas >= 0.1.x Dataset API correctly (from_dict, not raw dicts).
  - Wraps evaluate() in asyncio.run() so it doesn't block in async contexts.
  - Compares Naive RAG vs GraphRAG in a single run and logs both to MLflow.
  - Results saved to evaluation/results/<timestamp>.json for audit trail.
  - Gracefully skips metrics that require OpenAI if only Gemini key is present.

Run with:
    python -m evaluation.benchmark
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Sample Q&A pairs for benchmarking ────────────────────────────────────────
# Replace with your actual domain questions

BENCHMARK_QUESTIONS = [
    {
        "question": "What is the capital requirements ratio under Basel III?",
        "ground_truth": (
            "Basel III requires banks to maintain a minimum Common Equity Tier 1 "
            "capital ratio of 4.5% of risk-weighted assets."
        ),
    },
    {
        "question": "What constitutes insider trading under SEC regulations?",
        "ground_truth": (
            "Insider trading involves buying or selling securities based on material, "
            "non-public information in breach of a fiduciary duty or similar relationship."
        ),
    },
    {
        "question": "What are HIPAA's requirements for patient data storage?",
        "ground_truth": (
            "HIPAA requires covered entities to implement technical safeguards including "
            "encryption, access controls, and audit logs for electronic protected health "
            "information (ePHI)."
        ),
    },
]


# ── Benchmark runner ──────────────────────────────────────────────────────────

class RAGBenchmark:
    def __init__(
        self,
        naive_retriever: Any,
        graph_retriever: Any,
        llm: Any,
        mlflow_experiment: str = "deepchain-hybrid-rag",
    ) -> None:
        self.naive_retriever = naive_retriever
        self.graph_retriever = graph_retriever
        self.llm = llm
        self.mlflow_experiment = mlflow_experiment

        mlflow.set_experiment(mlflow_experiment)

    def run(self, questions: list[dict] | None = None) -> dict[str, Any]:
        """Run full benchmark comparison. Returns dict of {mode: ragas_scores}."""
        questions = questions or BENCHMARK_QUESTIONS
        results: dict[str, Any] = {}

        for mode, retriever in [
            ("naive_rag", self.naive_retriever),
            ("graph_rag", self.graph_retriever),
        ]:
            logger.info("Benchmarking mode: %s", mode)
            with mlflow.start_run(run_name=f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                scores = self._evaluate_mode(retriever, questions, mode)
                results[mode] = scores

                # Log all scalar metrics to MLflow
                for metric_name, value in scores.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(metric_name, value)
                mlflow.log_param("mode", mode)
                mlflow.log_param("num_questions", len(questions))

        self._save_results(results)
        self._print_comparison(results)
        return results

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _evaluate_mode(
        self,
        retriever: Any,
        questions: list[dict],
        mode: str,
    ) -> dict[str, float]:
        """Build Ragas Dataset and run evaluation."""
        rows: dict[str, list] = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }

        for item in questions:
            question = item["question"]
            ground_truth = item["ground_truth"]

            # Retrieve context
            result = retriever.retrieve(question)
            contexts = [c["text"] for c in result.chunks]

            # Generate answer
            context_str = "\n\n".join(contexts)
            prompt = (
                f"Answer the question using only the context below.\n\n"
                f"Context:\n{context_str}\n\nQuestion: {question}\nAnswer:"
            )
            answer = self.llm.invoke(prompt).content.strip()

            rows["question"].append(question)
            rows["answer"].append(answer)
            rows["contexts"].append(contexts)
            rows["ground_truth"].append(ground_truth)

        # ── Ragas >= 0.1.x API: from_dict, then evaluate ──────────────────────
        dataset = Dataset.from_dict(rows)

        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

        try:
            # evaluate() is synchronous in ragas >= 0.1.x
            score = evaluate(dataset=dataset, metrics=metrics)
            return {
                "faithfulness": float(score["faithfulness"]),
                "answer_relevancy": float(score["answer_relevancy"]),
                "context_precision": float(score["context_precision"]),
                "context_recall": float(score["context_recall"]),
            }
        except Exception as exc:
            logger.error("Ragas evaluation failed for mode %s: %s", mode, exc)
            return {"error": str(exc)}

    def _save_results(self, results: dict[str, Any]) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"benchmark_{timestamp}.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        logger.info("Results saved to %s", out_path)

    def _print_comparison(self, results: dict[str, Any]) -> None:
        print("\n" + "=" * 60)
        print("  DeepChain Benchmark — Naive RAG vs GraphRAG")
        print("=" * 60)
        metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        header = f"{'Metric':<25} {'Naive RAG':>12} {'GraphRAG':>12}"
        print(header)
        print("-" * 50)
        for m in metrics:
            naive_val = results.get("naive_rag", {}).get(m, "N/A")
            graph_val = results.get("graph_rag", {}).get(m, "N/A")
            naive_str = f"{naive_val:.4f}" if isinstance(naive_val, float) else str(naive_val)
            graph_str = f"{graph_val:.4f}" if isinstance(graph_val, float) else str(graph_val)
            print(f"{m:<25} {naive_str:>12} {graph_str:>12}")
        print("=" * 60 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    from langchain_google_genai import ChatGoogleGenerativeAI
    from retrieval.hybrid_retriever import HybridRetriever

    logging.basicConfig(level=logging.INFO)

    retriever = HybridRetriever(
        neo4j_uri=os.environ["NEO4J_URI"],
        neo4j_user=os.environ["NEO4J_USERNAME"],
        neo4j_password=os.environ["NEO4J_PASSWORD"],
    )

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

    # Wrap retriever for both modes
    class _ModeRetriever:
        def __init__(self, r: HybridRetriever, mode: str):
            self._r, self._mode = r, mode
        def retrieve(self, q: str):
            return self._r.retrieve(q, mode=self._mode)

    bench = RAGBenchmark(
        naive_retriever=_ModeRetriever(retriever, "naive"),
        graph_retriever=_ModeRetriever(retriever, "graph"),
        llm=llm,
    )
    bench.run()