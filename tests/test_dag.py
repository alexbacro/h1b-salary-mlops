"""The Airflow DAG loads and wires the pipeline in the right order.

Airflow is not in requirements-dev.txt (it runs in its own image), so CI skips these tests.
Run them wherever Airflow is installed, e.g. `pytest tests/test_dag.py` in an Airflow environment.
"""

from pathlib import Path

import pytest

pytest.importorskip("airflow")

from airflow.dag_processing.dagbag import DagBag  # noqa: E402  (import only after the skip check)

DAGS_DIR = Path(__file__).resolve().parents[1] / "dags"
EXPECTED_ORDER = ["ingest", "clean", "split", "baseline", "register_baseline", "train"]


@pytest.fixture(scope="module")
def dag():
    bag = DagBag(dag_folder=str(DAGS_DIR))  # only our dags/ folder is parsed
    assert bag.import_errors == {}
    # bag.dags holds what was parsed from the files; get_dag() would also query Airflow's database.
    return bag.dags.get("h1b_pipeline")


def test_dag_has_the_pipeline_tasks_in_order(dag):
    assert dag is not None
    assert set(dag.task_ids) == set(EXPECTED_ORDER)
    for upstream, downstream in zip(EXPECTED_ORDER, EXPECTED_ORDER[1:], strict=False):
        assert dag.get_task(downstream).upstream_task_ids == {upstream}


def test_dag_is_manual_and_does_not_backfill(dag):
    assert dag.schedule is None
    assert dag.catchup is False


def test_every_stage_uses_the_project_interpreter(dag):
    for task in dag.tasks:
        assert "-m h1b." in task.bash_command


def test_register_baseline_is_a_skippable_bootstrap_and_train_still_runs(dag):
    assert dag.get_task("register_baseline").skip_on_exit_code == [99]
    assert dag.get_task("train").trigger_rule == "none_failed"
    assert "--promote if-better" in dag.get_task("train").bash_command
