# Load MovieLens latest-small into BigQuery and copy ratings to the bucket.
# Phase 1 of the recommender: later phases read these tables and the CSV.
# Run from the repo root: python -m data.load_movielens
import io
import urllib.request
import zipfile

import pandas as pd
from google.cloud import bigquery, storage

import config

# GroupLens hosts this zip. It is the small public ratings set (~100k rows).
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def main() -> None:
    # Download the zip in memory. No local file to clean up.
    raw = urllib.request.urlopen(MOVIELENS_URL).read()
    archive = zipfile.ZipFile(io.BytesIO(raw))
    ratings = pd.read_csv(archive.open("ml-latest-small/ratings.csv"))
    movies = pd.read_csv(archive.open("ml-latest-small/movies.csv"))

    # WRITE_TRUNCATE replaces the table if we run this again.
    bq = bigquery.Client(project=config.PROJECT)
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    for name, frame in [("ratings", ratings), ("movies", movies)]:
        table_id = f"{config.PROJECT}.{config.BQ_DATASET}.{name}"
        bq.load_table_from_dataframe(frame, table_id, job_config=job_config).result()

    # The two-tower trainer reads this CSV from Cloud Storage, not BigQuery.
    # config.BUCKET is "gs://...", so [5:] is the bucket name.
    bucket_name = config.BUCKET[5:]
    storage.Client(project=config.PROJECT).bucket(bucket_name).blob(
        "raw/ratings.csv"
    ).upload_from_string(ratings.to_csv(index=False))

    print(
        len(ratings),
        "ratings |",
        len(movies),
        "movies |",
        ratings.userId.nunique(),
        "users",
    )


if __name__ == "__main__":
    main()
