# Two-tower retriever for MovieLens.
# Vertex runs this inside a TensorFlow container. It is packaged as the
# training task, so the body runs under `if __name__ == "__main__"`.
#
# A user tower and a movie tower each emit one vector. Movies a user liked
# (rating >= 4) are pulled close to that user. Every other movie in the
# batch is a negative. Recall@50 is the tuning metric: how often the user's
# newest held-out like appears in the top 50 nearest movies.
import argparse
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf


def _as_int(value: str) -> int:
    # Discrete hyperparameters arrive as floats, for example "64.0".
    return int(float(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dim", type=_as_int, default=64)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--epochs", type=_as_int, default=5)
    # 0 during tuning. The export job passes 1 and writes the vectors.
    parser.add_argument("--export", type=_as_int, default=0)
    return parser.parse_args()


def tower(vocab: list[str], dim: int) -> tf.keras.Sequential:
    # StringLookup maps an id to an index. The embedding is a fixed 64,
    # then a dense layer makes the vector length equal to the tuned dim.
    return tf.keras.Sequential(
        [
            tf.keras.layers.StringLookup(vocabulary=vocab),
            tf.keras.layers.Embedding(len(vocab) + 1, 64),
            tf.keras.layers.Dense(dim),
        ]
    )


class TwoTower(tf.keras.Model):
    """In-batch softmax: the matching movie is the positive, the rest are not."""

    def __init__(self, users: list[str], movies: list[str], dim: int):
        super().__init__()
        self.user = tower(users, dim)
        self.movie = tower(movies, dim)

    def train_step(self, data):
        user_ids, movie_ids = data
        with tf.GradientTape() as tape:
            user_vec = tf.math.l2_normalize(self.user(user_ids), axis=1)
            movie_vec = tf.math.l2_normalize(self.movie(movie_ids), axis=1)
            # Temperature 0.05 sharpens the softmax over the batch.
            logits = tf.matmul(user_vec, movie_vec, transpose_b=True) / 0.05
            labels = tf.range(tf.shape(logits)[0])
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    labels, logits, from_logits=True
                )
            )
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}


def recall_at_50(model: TwoTower, users, movies, test: pd.DataFrame) -> float:
    # Score every held-out user against every movie. Small data, so a
    # dense matrix multiply is enough. No approximate index yet.
    user_matrix = tf.math.l2_normalize(model.user(tf.constant(users)), axis=1).numpy()
    movie_matrix = tf.math.l2_normalize(model.movie(tf.constant(movies)), axis=1).numpy()
    user_index = {user_id: i for i, user_id in enumerate(users)}
    movie_index = {movie_id: i for i, movie_id in enumerate(movies)}
    query = user_matrix[[user_index[user_id] for user_id in test.userId]]
    top50 = np.argsort(-(query @ movie_matrix.T), axis=1)[:, :50]
    hits = [
        movie_index[movie_id] in row
        for movie_id, row in zip(test.movieId, top50)
    ]
    return float(np.mean(hits))


def export_embeddings(model: TwoTower, users, movies, out_uri: str) -> None:
    # Cloud Storage is mounted at /gcs inside the training container.
    user_matrix = tf.math.l2_normalize(model.user(tf.constant(users)), axis=1).numpy()
    movie_matrix = tf.math.l2_normalize(model.movie(tf.constant(movies)), axis=1).numpy()
    out = out_uri.replace("gs://", "/gcs/")
    batches = [
        ("movies", movies, movie_matrix, "id"),
        ("users", users, user_matrix, "user_id"),
    ]
    for folder, ids, vectors, key in batches:
        os.makedirs(f"{out}/{folder}", exist_ok=True)
        path = f"{out}/{folder}/{folder}.json"
        with open(path, "w") as handle:
            for item_id, vector in zip(ids, vectors):
                row = {key: item_id, "embedding": np.round(vector, 5).tolist()}
                handle.write(json.dumps(row) + "\n")


def main() -> None:
    args = parse_args()
    # The launcher passes a gs:// path. Training containers see that as /gcs/.
    ratings_path = args.ratings.replace("gs://", "/gcs/")
    frame = pd.read_csv(ratings_path)
    liked = frame[frame.rating >= 4].sort_values("timestamp")
    liked = liked.astype({"userId": str, "movieId": str})

    # Newest like per user is the test item. The rest train the towers.
    test = liked.groupby("userId").tail(1)
    train = liked.drop(test.index)
    users = sorted(liked.userId.unique())
    # Movie vocab is every movie that was rated, not only the liked ones.
    movies = sorted(frame.movieId.astype(str).unique())

    model = TwoTower(users, movies, args.dim)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr))
    dataset = (
        tf.data.Dataset.from_tensor_slices((train.userId.values, train.movieId.values))
        .shuffle(100_000)
        .batch(512)
    )
    model.fit(dataset, epochs=args.epochs, verbose=2)

    recall = recall_at_50(model, users, movies, test)
    print(f"recall@50 = {recall:.3f}")

    # Vertex reads this metric to pick the winning trial.
    import hypertune

    hypertune.HyperTune().report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag="recall_at_50",
        metric_value=recall,
        global_step=args.epochs,
    )

    if args.export:
        export_embeddings(model, users, movies, args.out)


if __name__ == "__main__":
    main()
