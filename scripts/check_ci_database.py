"""Fail quickly when the CI database is missing its corpus, schema, or grants."""
from sqlalchemy import text

from app.database import SessionLocal
from app.evaluation.experiment_runner import validate_benchmark
from app.evaluation.heldout_benchmark import HELDOUT_BENCHMARK


REQUIRED_PRIVILEGES = {
    "documents": ("SELECT",),
    "query_logs": ("SELECT", "INSERT", "UPDATE"),
    "experiments": ("SELECT", "INSERT", "UPDATE"),
    "experiment_results": ("SELECT", "INSERT", "UPDATE"),
}
WRITE_TABLES = ("query_logs", "experiments", "experiment_results")


def main() -> None:
    with SessionLocal() as db:
        missing = []
        for table, privileges in REQUIRED_PRIVILEGES.items():
            for privilege in privileges:
                allowed = db.execute(
                    text("SELECT has_table_privilege(current_user, :table, :privilege)"),
                    {"table": table, "privilege": privilege},
                ).scalar_one()
                if not allowed:
                    missing.append(f"{privilege} on {table}")
        if missing:
            raise RuntimeError("CI database role is missing: " + ", ".join(missing))

        for table in WRITE_TABLES:
            sequence = db.execute(
                text("SELECT pg_get_serial_sequence(:table, 'id')"),
                {"table": table},
            ).scalar_one()
            if sequence and not db.execute(
                text("SELECT has_sequence_privilege(current_user, :sequence, 'USAGE')"),
                {"sequence": sequence},
            ).scalar_one():
                missing.append(f"USAGE on {sequence}")
        if missing:
            raise RuntimeError("CI database role is missing: " + ", ".join(missing))

        document_count = db.execute(text("SELECT count(*) FROM documents")).scalar_one()
        validate_benchmark(db, HELDOUT_BENCHMARK)
        print(f"CI database ready: {document_count} documents and held-out labels verified")


if __name__ == "__main__":
    main()
