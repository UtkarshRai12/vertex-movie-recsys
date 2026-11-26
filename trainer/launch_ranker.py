# Train the ranker on Vertex and, on the first run, deploy it.
# The endpoint bills by the hour until the model is undeployed.
# Run from the repo root: python -m trainer.launch_ranker
import argparse
import json
from pathlib import Path

from google.cloud import aiplatform

import config

TRAIN_IMAGE = "us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-0:latest"
# sklearn 1.5 serving image. The training image is 1.0, so requirements
# below upgrade scikit-learn before the script runs.
SERVE_IMAGE = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest"
ENDPOINT_PATH = Path(__file__).with_name("ranker_endpoint.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", default="200")
    parser.add_argument("--max_depth", default="3")
    # Set this to add a new version. Leave it empty to create the model.
    parser.add_argument("--parent_model")
    # Deploy onto this endpoint instead of creating a new one.
    parser.add_argument("--endpoint")
    args = parser.parse_args()

    aiplatform.init(
        project=config.PROJECT,
        location=config.REGION,
        staging_bucket=config.BUCKET,
    )
    job = aiplatform.CustomTrainingJob(
        display_name="ranker",
        script_path="trainer/ranker.py",
        container_uri=TRAIN_IMAGE,
        # Train with the same sklearn the 1.5 serving image loads.
        # numpy 1.23.5 is one that image can unpickle.
        requirements=[
            "google-cloud-bigquery",
            "google-cloud-storage",
            "db-dtypes",
            "numpy==1.23.5",
            "scikit-learn==1.5.2",
        ],
        model_serving_container_image_uri=SERVE_IMAGE,
    )
    model = job.run(
        args=[
            "--project",
            config.PROJECT,
            "--n_estimators",
            args.n_estimators,
            "--max_depth",
            args.max_depth,
        ],
        machine_type="n1-standard-4",
        model_display_name="ranker",
        parent_model=args.parent_model,
        service_account=f"p2-trainer@{config.PROJECT}.iam.gserviceaccount.com",
    )
    print("model:", model.resource_name, "version:", model.version_id)

    if args.endpoint:
        endpoint = aiplatform.Endpoint(args.endpoint)
        model.deploy(
            endpoint=endpoint,
            machine_type="n1-standard-2",
            min_replica_count=1,
            max_replica_count=3,
        )
    elif args.parent_model:
        return
    else:
        endpoint = model.deploy(
            machine_type="n1-standard-2",
            min_replica_count=1,
            max_replica_count=3,
        )
    print("RANKER_ENDPOINT =", endpoint.resource_name)
    prediction = endpoint.predict(instances=[[4.2, 300, 4.1, 250, 0.62, 1.0]])
    print(prediction.predictions)
    ENDPOINT_PATH.write_text(
        json.dumps(
            {
                "model": model.resource_name,
                "version": model.version_id,
                "endpoint": endpoint.resource_name,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
