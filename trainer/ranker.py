# Rank movies by the star rating a user would give them.
# Vertex runs this inside the scikit-learn training container.
# The saved model is what the matching prediction image loads.
import argparse
import os
import sys

import joblib
import numpy as np
import sklearn
from google.cloud import bigquery
from google.cloud import storage
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Serving sends one list per candidate, in this exact order.
FEATURES = ["u_avg", "u_n", "m_avg", "m_n", "sim", "genre_match"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=3)
    args = parser.parse_args()

    frame = bigquery.Client(project=args.project).query(
        f"SELECT * FROM `{args.project}.movielens.rank_train`"
    ).to_dataframe()
    train_x, test_x, train_y, test_y = train_test_split(
        frame[FEATURES].values,
        frame.rating.values,
        test_size=0.2,
        random_state=42,
    )
    model = GradientBoostingRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    ).fit(train_x, train_y)
    rmse = mean_squared_error(test_y, model.predict(test_x)) ** 0.5
    print("python", sys.version)
    print("sklearn", sklearn.__version__)
    print("numpy", np.__version__)
    print("test RMSE:", round(rmse, 3))

    # Protocol 4 is what the sklearn 1.0 serving image can unpickle.
    # Writing through the /gcs mount produced a file that image rejected.
    local_path = "/tmp/model.joblib"
    joblib.dump(model, local_path, protocol=4)
    uri = os.environ["AIP_MODEL_DIR"]
    bucket_name, _, prefix = uri[len("gs://") :].partition("/")
    blob_name = prefix.rstrip("/") + "/model.joblib"
    storage.Client(project=args.project).bucket(bucket_name).blob(
        blob_name
    ).upload_from_filename(local_path)
    print("wrote", f"gs://{bucket_name}/{blob_name}", os.path.getsize(local_path))


if __name__ == "__main__":
    main()
