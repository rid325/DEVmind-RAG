import unittest
from contextlib import contextmanager
from types import SimpleNamespace as NS
from unittest.mock import patch

from scripts import run_pr_evaluation as evaluation


class PrEvaluationSessionTests(unittest.TestCase):
    def test_long_experiment_runs_between_short_database_sessions(self):
        active_sessions = 0
        session_count = 0
        experiments = {}

        class FakeDb:
            def get(self, _model, experiment_id):
                return experiments[experiment_id]

        @contextmanager
        def sessions():
            nonlocal active_sessions, session_count
            active_sessions += 1
            session_count += 1
            try:
                yield FakeDb()
            finally:
                active_sessions -= 1

        def create_experiment(_db, request):
            experiment_id = len(experiments) + 1
            experiment = NS(id=experiment_id, name=request.name, status="queued", error=None)
            experiments[experiment_id] = experiment
            return experiment

        def run_experiment(experiment_id):
            self.assertEqual(active_sessions, 0)
            experiments[experiment_id].status = "complete"

        comparison = {"gates": {"passed": True}}
        with patch("app.database.SessionLocal", sessions), \
             patch("app.evaluation.experiment_runner.create_experiment", create_experiment), \
             patch("app.evaluation.experiment_runner.run_experiment", run_experiment), \
             patch("app.evaluation.comparator.compare_experiments", return_value=comparison), \
             patch.object(evaluation, "_run_summary", side_effect=lambda _db, item: {"id": item.id}), \
             patch.object(evaluation, "find_unjudged_full_top5", return_value=[]):
            report = evaluation.run_evaluation(8000)

        self.assertEqual(session_count, 5)
        self.assertEqual(report["comparison"], comparison)
        self.assertEqual(report["runs"], [{"id": 1}, {"id": 2}])


if __name__ == "__main__":
    unittest.main()
