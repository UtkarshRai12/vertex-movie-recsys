# One final two-tower run that writes user and movie embeddings to the bucket.
# Uses trainer/best_params.json from the tuning job unless you pass overrides.
# Run from the repo root: python -m trainer.export
import argparse
import json
from pathlib import Path

from google.cloud import aiplatform

import config

BEST_PARAMS_PATH = Path(__file__).with_name("best_params.json")


def main() -> None:
    saved = {}
    if BEST_PARAMS_PATH.exists():
        saved = json.loads(BEST_PARAMS_PATH.read_text())

    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", default=saved.get("dim"))
    parser.add_argument("--lr", default=saved.get("lr"))
    args = parser.parse_args()
    if args.dim is None or args.lr is None:
        raise SystemExit("No best params yet. Run python -m trainer.tune first.")

    aiplatform.init(
        project=config.PROJECT,
        location=config.REGION,
        staging_bucket=config.BUCKET,
    )
    job = aiplatform.CustomJob.from_local_script(
        display_name="two-tower-export",
        script_path="trainer/two_tower.py",
        container_uri="us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-15.py310:latest",
        requirements=["cloudml-hypertune", "pandas"],
        args=[
            "--ratings",
            f"{config.BUCKET}/raw/ratings.csv",
            "--out",
            f"{config.BUCKET}/emb",
            "--dim",
            str(int(float(args.dim))),
            "--lr",
            str(args.lr),
            "--export",
            "1",
        ],
        machine_type="n1-standard-8",
    )
    job.run(service_account=f"p2-trainer@{config.PROJECT}.iam.gserviceaccount.com")
    print("exported to", f"{config.BUCKET}/emb")


if __name__ == "__main__":
    main()
