"""Run the held-out baseline/full comparison and write CI-friendly reports."""
from argparse import ArgumentParser
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback


METRIC_LABELS = {
    "retrieval_recall": "Pooled Recall@5",
    "retrieval_precision": "Precision@5",
    "mrr": "MRR",
    "ndcg": "NDCG@5",
    "faithfulness_score": "Faithfulness",
    "latency_ms": "Latency",
}


def _number(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def _run_summary(db, experiment):
    from app.models import ExperimentResult

    rows = db.query(ExperimentResult).filter_by(experiment_id=experiment.id).all()
    failures = [
        {"question_id": row.benchmark_id, "status": row.status, "error": row.error}
        for row in rows if row.status != "complete"
    ]
    return {
        "id": experiment.id,
        "name": experiment.name,
        "status": experiment.status,
        "result_count": len(rows),
        "failures": failures,
    }


def render_markdown(report):
    comparison = report["comparison"]
    metrics = comparison["metrics"]
    gates = comparison["gates"]
    icon = "✅" if gates["passed"] else "❌"
    lines = [
        "<!-- devmind-rag-evaluation -->",
        f"## {icon} RAG evaluation gate",
        "",
        "| Metric | Baseline | Full pipeline | Delta | Gate |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for key in ("retrieval_recall", "retrieval_precision", "mrr", "ndcg", "faithfulness_score"):
        values = metrics[key]
        gate = gates["quality"][key]
        lines.append(
            f"| {METRIC_LABELS[key]} | {_number(values['mean_a'])} | "
            f"{_number(values['mean_b'])} | {_number(values['difference_b_minus_a'])} | "
            f"{gate['status']} (Holm p={_number(gate['adjusted_p_value'])}) |"
        )
    latency = metrics["latency_ms"]
    latency_gate = gates["latency"]
    lines.append(
        f"| Full p95 latency | {_number(latency['p95_a'], 0)} ms | "
        f"{_number(latency_gate['p95_b_ms'], 0)} ms | "
        f"{_number(latency['difference_b_minus_a'], 0)} ms mean | "
        f"{latency_gate['status']} (ceiling {_number(latency_gate['ceiling_ms'], 0)} ms) |"
    )
    lines.extend([
        "",
        f"**Overall: {'passed' if gates['passed'] else 'failed'}.**",
        "",
        "A positive direction across metrics is evidence worth following up, but with 20 questions "
        "the sample may not have enough power to confirm an effect. A nonsignificant result means "
        "the run did not establish a difference; it does not establish that there is no difference.",
    ])

    unjudged = report["unjudged_full_top5"]
    if unjudged:
        lines.extend([
            "",
            f"The full pipeline returned **{len(unjudged)} unjudged top-5 result(s)** outside the frozen pool. "
            "They receive zero gain by construction, so the full-pipeline ranking scores may be slightly "
            "underestimated. Review and version the pool before treating them as irrelevant.",
        ])

    failures = report["runs"][1]["failures"]
    if failures:
        timeout_count = sum(item["error"] == "APITimeoutError" for item in failures)
        lines.extend([
            "",
            f"The full pipeline recorded **{len(failures)} failed measurement(s)**"
            + (f", including {timeout_count} API timeout(s)." if timeout_count else "."),
        ])
    lines.extend(["", "The workflow artifact contains the complete aggregate JSON report."])
    return "\n".join(lines) + "\n"


def find_unjudged_full_top5(db, full_experiment):
    from app.models import ExperimentResult

    benchmark = {item["id"]: item for item in full_experiment.benchmark_snapshot}
    rows = db.query(ExperimentResult).filter_by(experiment_id=full_experiment.id).all()
    missing = []
    for row in rows:
        judged = {int(item[0]) for item in benchmark[row.benchmark_id]["judgments"]}
        for document_id in row.metric_details.get("retrieved_document_ids", [])[:5]:
            if document_id not in judged:
                missing.append({"question_id": row.benchmark_id, "document_id": document_id})
    return missing


def run_evaluation(latency_ceiling_ms):
    from app.database import SessionLocal
    from app.evaluation.comparator import compare_experiments
    from app.evaluation.experiment_runner import ExperimentRequest, create_experiment, run_experiment

    stamp = os.getenv("GITHUB_RUN_ID", datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"))
    revision = os.getenv("PR_HEAD_SHA", "local")
    configs = (
        ("baseline", {"use_hyde": False, "use_reranking": False, "use_expansion": False}),
        ("full", {"use_hyde": True, "use_reranking": True, "use_expansion": True}),
    )
    experiments = []
    with SessionLocal() as db:
        for label, config in configs:
            experiment = create_experiment(db, ExperimentRequest(
                name=f"pr-{stamp}-{label}", benchmark="heldout_20", config=config,
                description=f"GitHub Actions baseline-versus-full release gate at {revision}",
            ))
            run_experiment(experiment.id)
            db.expire_all()
            experiment = db.get(type(experiment), experiment.id)
            if experiment.status not in {"complete", "complete_with_errors"}:
                raise RuntimeError(f"{label} experiment ended with status {experiment.status}: {experiment.error}")
            experiments.append(experiment)

        comparison = compare_experiments(
            db, experiments[0].id, experiments[1].id, latency_ceiling_ms
        )
        return {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "revision": revision,
            "comparison": comparison,
            "runs": [_run_summary(db, item) for item in experiments],
            "unjudged_full_top5": find_unjudged_full_top5(db, experiments[1]),
        }


def main():
    parser = ArgumentParser()
    parser.add_argument("--json", default="rag-evaluation.json")
    parser.add_argument("--markdown", default="rag-evaluation.md")
    parser.add_argument("--latency-ceiling-ms", type=float, default=8000.0)
    args = parser.parse_args()
    json_path, markdown_path = Path(args.json), Path(args.markdown)
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not configured")
        if not os.getenv("DATABASE_URL"):
            raise RuntimeError("CI_DATABASE_URL is not configured")
        report = run_evaluation(args.latency_ceiling_ms)
        json_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
        markdown_path.write_text(render_markdown(report))
        return 0 if report["comparison"]["gates"]["passed"] else 1
    except Exception as exc:
        failure = {"recorded_at": datetime.now(timezone.utc).isoformat(),
                   "error": type(exc).__name__, "message": str(exc)}
        json_path.write_text(json.dumps(failure, indent=2) + "\n")
        markdown_path.write_text(
            "<!-- devmind-rag-evaluation -->\n## ❌ RAG evaluation gate\n\n"
            f"The evaluation could not complete: `{type(exc).__name__}: {exc}`\n"
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
