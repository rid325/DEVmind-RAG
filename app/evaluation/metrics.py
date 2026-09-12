"""Small, explicit metrics. Keyword coverage is a proxy, not semantic relevance."""
import math
import re
from statistics import mean, stdev

import numpy as np
from scipy.stats import ttest_rel

METRICS = ("retrieval_recall", "retrieval_precision", "mrr", "ndcg",
           "faithfulness_score", "answer_relevance", "latency_ms")
QUALITY_GATE_METRICS = ("retrieval_recall", "retrieval_precision", "mrr", "ndcg",
                        "faithfulness_score")
DEFAULT_LATENCY_P95_CEILING_MS = 8000.0


def keyword_coverage(answer: str, keywords: list[str]) -> float | None:
    if not keywords:
        return None
    normalized = " " + " ".join(re.findall(r"\w+", answer.lower())) + " "
    def matches(group):
        return any(
            " " + " ".join(re.findall(r"\w+", alternative.lower())) + " " in normalized
            for alternative in group.split("/") if alternative.strip()
        )
    return sum(matches(group) for group in keywords) / len(keywords)


def annotated_recall(retrieved_ids: list[int], relevant_ids: list[int]) -> float | None:
    relevant = set(relevant_ids)
    return len(set(retrieved_ids) & relevant) / len(relevant) if relevant else None


def ranking_metrics(retrieved_ids: list[int], judgments: dict[int, int], k: int = 5) -> dict:
    """Binary P/R/MRR and graded NDCG for an ordered result list.

    A grade above zero is relevant for the binary metrics. Recall is bounded by
    the judged pool and must not be described as exhaustive corpus recall.
    """
    if k <= 0 or not judgments:
        return {"retrieval_precision": None, "retrieval_recall": None,
                "mrr": None, "ndcg": None}
    ranked = retrieved_ids[:k]
    relevant = {document_id for document_id, grade in judgments.items() if grade > 0}
    precision = sum(judgments.get(document_id, 0) > 0 for document_id in ranked) / k
    recall = (sum(document_id in relevant for document_id in set(ranked)) / len(relevant)
              if relevant else None)
    first = next((rank for rank, document_id in enumerate(ranked, 1)
                  if judgments.get(document_id, 0) > 0), None)
    mrr = 1 / first if first else 0.0
    gains = [judgments.get(document_id, 0) for document_id in ranked]
    dcg = sum((2 ** grade - 1) / math.log2(rank + 1)
              for rank, grade in enumerate(gains, 1))
    ideal = sorted(judgments.values(), reverse=True)[:k]
    idcg = sum((2 ** grade - 1) / math.log2(rank + 1)
               for rank, grade in enumerate(ideal, 1))
    return {"retrieval_precision": precision, "retrieval_recall": recall,
            "mrr": mrr, "ndcg": dcg / idcg if idcg else None}


def metric_value(row, metric: str):
    value = getattr(row, metric, None)
    if value is None and metric in {"retrieval_precision", "mrr", "ndcg"}:
        value = (getattr(row, "metric_details", None) or {}).get(metric)
    return value


def compare_results(rows_a, rows_b) -> dict:
    """Pair by benchmark ID and query; drop missing values independently per metric."""
    b_by_id = {row.benchmark_id: row for row in rows_b}
    paired = [(a, b_by_id[a.benchmark_id]) for a in rows_a
              if a.benchmark_id in b_by_id and a.query == b_by_id[a.benchmark_id].query]
    result = {}
    for metric in METRICS:
        values = [(metric_value(a, metric), metric_value(b, metric)) for a, b in paired
                  if a.status != "failed" and b.status != "failed"]
        values = [(a, b) for a, b in values
                  if a is not None and b is not None and math.isfinite(a) and math.isfinite(b)]
        n = len(values)
        a, b = zip(*values) if values else ([], [])
        avg_a, avg_b = (mean(a), mean(b)) if n else (None, None)
        difference = avg_b - avg_a if n else None
        improvement = -difference if n and metric == "latency_ms" else difference
        p_value, note = None, None
        if n < 2:
            note = "At least two complete pairs are needed."
        else:
            differences = [y - x for x, y in values]
            if all(d == 0 for d in differences):
                p_value = 1.0
            elif stdev(differences) == 0:
                # The t statistic tends to +/- infinity when every nonzero
                # paired difference is identical, giving a limiting p-value 0.
                p_value = 0.0
                note = "Every paired difference is the same nonzero value."
            else:
                p = float(ttest_rel(a, b).pvalue)
                p_value = p if math.isfinite(p) else None
        result[metric] = {
            "paired_count": n, "excluded_pairs": len(paired) - n,
            "mean_a": avg_a, "std_a": stdev(a) if n >= 2 else None,
            "mean_b": avg_b, "std_b": stdev(b) if n >= 2 else None,
            "p95_a": float(np.percentile(a, 95)) if n else None,
            "p95_b": float(np.percentile(b, 95)) if n else None,
            "difference_b_minus_a": difference,
            "improvement_percent": 100 * improvement / abs(avg_a) if n and avg_a != 0 else None,
            "p_value": p_value, "significant": p_value is not None and p_value < 0.05,
            "b_is_better": improvement > 0 if improvement is not None else None,
            "note": note,
        }
    # Holm correction across metrics that produced a finite p-value.
    ranked = sorted((v["p_value"], key) for key, v in result.items() if v["p_value"] is not None)
    adjusted = 0.0
    for rank, (p, key) in enumerate(ranked):
        adjusted = max(adjusted, min(1.0, (len(ranked) - rank) * p))
        result[key]["p_value_holm"] = adjusted
    for values in result.values():
        values.setdefault("p_value_holm", None)
        p = values["p_value_holm"]
        values["significant_holm"] = p is not None and p < 0.05
    return result


def evaluate_gates(comparison: dict, latency_p95_ceiling_ms: float = DEFAULT_LATENCY_P95_CEILING_MS,
                   alpha: float = 0.05) -> dict:
    """Fail significant quality regressions and any latency p95 ceiling breach."""
    metrics = comparison["metrics"]
    available = [(metrics[name]["p_value"], name) for name in QUALITY_GATE_METRICS
                 if name in metrics and metrics[name]["p_value"] is not None]
    ranked = sorted(available)
    adjusted = 0.0
    adjusted_by_name = {}
    for rank, (p_value, name) in enumerate(ranked):
        adjusted = max(adjusted, min(1.0, (len(available) - rank) * p_value))
        adjusted_by_name[name] = adjusted

    quality = {}
    for name in QUALITY_GATE_METRICS:
        values = metrics.get(name)
        if not values or values["paired_count"] < 2 or values["p_value"] is None:
            quality[name] = {"status": "insufficient_data", "passed": False,
                             "difference_b_minus_a": None, "adjusted_p_value": None}
            continue
        adjusted_p = adjusted_by_name[name]
        regression = values["difference_b_minus_a"] < 0 and adjusted_p < alpha
        quality[name] = {"status": "failed" if regression else "passed",
                         "passed": not regression,
                         "difference_b_minus_a": values["difference_b_minus_a"],
                         "adjusted_p_value": adjusted_p}

    latency = metrics.get("latency_ms")
    p95 = latency["p95_b"] if latency and latency["paired_count"] else None
    latency_passed = p95 is not None and p95 <= latency_p95_ceiling_ms
    latency_gate = {"status": "passed" if latency_passed else "failed" if p95 is not None else "insufficient_data",
                    "passed": latency_passed, "p95_b_ms": p95,
                    "ceiling_ms": latency_p95_ceiling_ms}
    return {"passed": all(item["passed"] for item in quality.values()) and latency_passed,
            "alpha": alpha, "quality_p_value_adjustment": "Holm across available quality gate metrics",
            "quality": quality, "latency": latency_gate}
