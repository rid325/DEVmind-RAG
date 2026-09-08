"""Small, explicit metrics. Keyword coverage is a proxy, not semantic relevance."""
import math
import re
from statistics import mean, stdev

from scipy.stats import ttest_rel

METRICS = ("retrieval_recall", "faithfulness_score", "answer_relevance", "latency_ms")


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


def compare_results(rows_a, rows_b) -> dict:
    """Pair by benchmark ID and query; drop missing values independently per metric."""
    b_by_id = {row.benchmark_id: row for row in rows_b}
    paired = [(a, b_by_id[a.benchmark_id]) for a in rows_a
              if a.benchmark_id in b_by_id and a.query == b_by_id[a.benchmark_id].query]
    result = {}
    for metric in METRICS:
        values = [(getattr(a, metric), getattr(b, metric)) for a, b in paired
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
                note = "The paired differences have zero variance; the t-test is undefined."
            else:
                p = float(ttest_rel(a, b).pvalue)
                p_value = p if math.isfinite(p) else None
        result[metric] = {
            "paired_count": n, "excluded_pairs": len(paired) - n,
            "mean_a": avg_a, "std_a": stdev(a) if n >= 2 else None,
            "mean_b": avg_b, "std_b": stdev(b) if n >= 2 else None,
            "difference_b_minus_a": difference,
            "improvement_percent": 100 * improvement / abs(avg_a) if n and avg_a != 0 else None,
            "p_value": p_value, "significant": p_value is not None and p_value < 0.05,
            "b_is_better": improvement > 0 if improvement is not None else None,
            "note": note,
        }
    # Holm correction for the family of four reported metrics.
    ranked = sorted((v["p_value"], key) for key, v in result.items() if v["p_value"] is not None)
    adjusted = 0.0
    for rank, (p, key) in enumerate(ranked):
        adjusted = max(adjusted, min(1.0, (len(METRICS) - rank) * p))
        result[key]["p_value_holm"] = adjusted
    for values in result.values():
        values.setdefault("p_value_holm", None)
        p = values["p_value_holm"]
        values["significant_holm"] = p is not None and p < 0.05
    return result
