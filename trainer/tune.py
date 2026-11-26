# Submit an 8-trial Vertex hyperparameter tuning job for the two-tower model.
# Run from the repo root: python -m trainer.tune
#
# Trials bill only while the machines are up. Two trials run at a time.
# The script blocks until every trial finishes, then prints the best dim and lr.
import json
from pathlib import Path

from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt

import config

# Written next to this file so the export job can reuse the winning values.
BEST_PARAMS_PATH = Path(__file__).with_name("best_params.json")


def _number(parameter) -> float:
    # The SDK usually returns a plain float. Older responses are protobuf Values.
    value = parameter.value
    if isinstance(value, (int, float)):
        return float(value)
    return float(value.number_value)


def main() -> None:
    aiplatform.init(
        project=config.PROJECT,
        location=config.REGION,
        staging_bucket=config.BUCKET,
    )
    job = aiplatform.CustomJob.from_local_script(
        display_name="two-tower",
        script_path="trainer/two_tower.py",
        container_uri="us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-15.py310:latest",
        requirements=["cloudml-hypertune", "pandas"],
        args=[
            "--ratings",
            f"{config.BUCKET}/raw/ratings.csv",
            "--out",
            f"{config.BUCKET}/emb",
        ],
        machine_type="n1-standard-8",
    )
    tuning_job = aiplatform.HyperparameterTuningJob(
        display_name="two-tower-hpt",
        custom_job=job,
        metric_spec={"recall_at_50": "maximize"},
        parameter_spec={
            "lr": hpt.DoubleParameterSpec(min=0.005, max=0.2, scale="log"),
            # Discrete specs need a scale. Linear is the right one for a short list.
            "dim": hpt.DiscreteParameterSpec(values=[32, 64, 128], scale="linear"),
        },
        max_trial_count=8,
        parallel_trial_count=2,
    )
    tuning_job.run(
        service_account=f"p2-trainer@{config.PROJECT}.iam.gserviceaccount.com"
    )

    finished = [
        trial
        for trial in tuning_job.trials
        if trial.final_measurement.metrics
    ]
    best = max(finished, key=lambda trial: trial.final_measurement.metrics[0].value)
    params = {item.parameter_id: _number(item) for item in best.parameters}
    recall = best.final_measurement.metrics[0].value
    # dim is a discrete float such as 64.0. Store it as an int for the export job.
    payload = {
        "dim": int(params["dim"]),
        "lr": params["lr"],
        "recall_at_50": recall,
    }
    BEST_PARAMS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print("best:", payload)


if __name__ == "__main__":
    main()
