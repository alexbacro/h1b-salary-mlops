"""Airflow DAG: the whole h1b pipeline, from DOL files to a promoted model.

    ingest -> clean -> split -> baseline -> register_baseline -> train

Every task runs the same `python -m h1b.<stage>` command as the Makefile, using the project's own
virtualenv in the Airflow image (docker/airflow.Dockerfile). Airflow's interpreter only schedules;
no pipeline logic lives here, so `make data` and this DAG can never disagree.

- register_baseline is a bootstrap: it runs only when no model holds the serving alias yet. On later
  runs it is skipped, because register_baseline always moves the alias and would demote a better model.
- train registers a FLAML model and promotes it only if its validation log-RMSE beats the champion's.

Triggered manually (no schedule): DOL publishes new files quarterly, and they are downloaded by hand.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

# The project's interpreter inside the Airflow image. Overridable so the DAG can be tested elsewhere.
H1B_PYTHON = os.environ.get("H1B_PYTHON", "/opt/h1b-venv/bin/python")
H1B_ROOT = os.environ.get("H1B_ROOT", "/opt/h1b")

# Exit code that tells BashOperator to mark the task "skipped" instead of "failed".
SKIP_EXIT_CODE = 99

# Exits 0 if no model holds the serving alias (so the baseline should be registered), 99 if one does.
# Any other error (e.g. MLflow unreachable) exits non-zero and fails the task instead of skipping it.
NO_CHAMPION_CHECK = f"""import sys
from mlflow import MlflowClient
from h1b.config import load_params

serving = load_params()["serving"]
models = MlflowClient().search_registered_models(filter_string=f"name='{{serving['model_name']}}'")
has_champion = bool(models) and serving["model_alias"] in models[0].aliases
print("champion exists:", has_champion)
sys.exit({SKIP_EXIT_CODE} if has_champion else 0)"""

# Check first; register only if the check exits 0. (`&&` must sit on the heredoc's first line: the
# closing EOF has to be alone on its own line.)
REGISTER_IF_NO_CHAMPION = f"""{H1B_PYTHON} - <<'EOF' && {H1B_PYTHON} -m h1b.register_baseline
{NO_CHAMPION_CHECK}
EOF"""


def stage(task_id: str, command: str, **kwargs) -> BashOperator:
    """A task that runs one pipeline stage with the project's interpreter, from the project root."""
    return BashOperator(task_id=task_id, bash_command=command, cwd=H1B_ROOT, **kwargs)


with DAG(
    dag_id="h1b_pipeline",
    description="DOL LCA files -> clean data -> baseline -> FLAML model promoted to @champion if better",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=2)},
    tags=["h1b", "mlops"],
    doc_md=__doc__,
) as dag:
    ingest = stage("ingest", f"{H1B_PYTHON} -m h1b.ingest")
    clean = stage("clean", f"{H1B_PYTHON} -m h1b.clean")
    split = stage("split", f"{H1B_PYTHON} -m h1b.split")
    baseline = stage("baseline", f"{H1B_PYTHON} -m h1b.baseline")
    register_baseline = stage(
        "register_baseline",
        REGISTER_IF_NO_CHAMPION,
        skip_on_exit_code=SKIP_EXIT_CODE,
    )
    train = stage(
        "train",
        f"{H1B_PYTHON} -m h1b.train --promote if-better",
        # Run after register_baseline whether it registered (success) or was skipped; stop if it failed.
        trigger_rule="none_failed",
    )

    ingest >> clean >> split >> baseline >> register_baseline >> train
