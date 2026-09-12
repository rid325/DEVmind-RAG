from app.models import Experiment, ExperimentResult
from app.evaluation.experiment_runner import FINISHED
from app.evaluation.metrics import compare_results, evaluate_gates


def compare_experiments(db, exp_a: int, exp_b: int, latency_p95_ceiling_ms: float = 8000.0) -> dict:
    a, b = db.get(Experiment, exp_a), db.get(Experiment, exp_b)
    if a is None or b is None:
        raise LookupError("Experiment not found")
    if exp_a == exp_b:
        raise ValueError("Choose two different experiments")
    if a.status not in FINISHED or b.status not in FINISHED:
        raise ValueError("Both experiments must finish successfully before comparison")
    if a.benchmark_version != b.benchmark_version or a.benchmark_snapshot != b.benchmark_snapshot:
        raise ValueError("Experiments used different benchmarks")
    if a.corpus_fingerprint != b.corpus_fingerprint:
        raise ValueError("Experiments used different corpus snapshots")
    rows_a = db.query(ExperimentResult).filter_by(experiment_id=exp_a).all()
    rows_b = db.query(ExperimentResult).filter_by(experiment_id=exp_b).all()
    comparison = {
        "experiment_a": {"id": a.id, "name": a.name, "config": a.config, "status": a.status},
        "experiment_b": {"id": b.id, "name": b.name, "config": b.config, "status": b.status},
        "benchmark_version": a.benchmark_version, "corpus_fingerprint": a.corpus_fingerprint,
        "metrics": compare_results(rows_a, rows_b),
        "notes": [
            "Development-set recall uses known supporting sources; held-out recall is bounded by its pooled judgments.",
            "Precision@5 and MRR treat grades 1 and 2 as relevant; NDCG@5 uses the full 0/1/2 grades.",
            "Answer relevance is keyword coverage, not an LLM or semantic relevance score.",
            "Latency excludes faithfulness judging and one-time model loading; lower is better.",
            "Means and sample standard deviations use the same non-missing pairs per metric.",
            "P-values are two-sided paired t-tests; quality gates use Holm adjustment across their available metrics.",
            "This corpus-grounded development benchmark is exploratory, not a held-out evaluation.",
        ],
    }
    comparison["gates"] = evaluate_gates(comparison, latency_p95_ceiling_ms)
    return comparison
