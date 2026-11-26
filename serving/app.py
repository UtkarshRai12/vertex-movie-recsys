# Real-time recommender. One request does three timed stages:
# Feature Store read, Vector Search, then the ranker.
# A user missing from the store follows the same last two stages. Their
# vector is the average of movies they already rated (see cold_user.py).
# With no ratings at all, we return the most-rated movies.
# The movie table is small and static, so it is loaded once at startup.
import os
import time

from fastapi import FastAPI
from google.api_core import exceptions as gax
from google.cloud import aiplatform, bigquery
from vertexai.resources.preview.feature_store import FeatureView

from cold_user import popular_items, stand_in

PROJECT = os.environ["PROJECT"]
REGION = "us-central1"
aiplatform.init(project=PROJECT, location=REGION)
user_view = FeatureView("user_features", feature_online_store_id="recsys_online")
index_ep = aiplatform.MatchingEngineIndexEndpoint(os.environ["INDEX_ENDPOINT"])
ranker = aiplatform.Endpoint(os.environ["RANKER_ENDPOINT"])
bq = bigquery.Client(project=PROJECT)
# movie_id is a string, matching the ids Vector Search returns.
MOVIES = (
    bq.query(f"SELECT * FROM `{PROJECT}.movielens.movie_stats`")
    .to_dataframe()
    .set_index("movie_id")
    .to_dict("index")
)
# Used only when the feature store has no embedding for this user.
# Loaded once. A request never writes this back.
EMBEDDINGS = {
    str(row.id): list(row.embedding)
    for row in bq.query(
        f"SELECT CAST(id AS STRING) AS id, embedding "
        f"FROM `{PROJECT}.movielens.movie_embeddings`"
    )
}
app = FastAPI()


def read_user(uid: str) -> dict:
    # Each feature value is wrapped, for example {"double_value": 4.1}.
    # Embeddings are {"double_array_value": {"values": [...]}}.
    # A missing key is a 404 from the store. That is a new user, not a failure.
    try:
        data = user_view.read(key=[uid]).to_dict()
    except gax.NotFound:
        return {}
    out = {}
    for feature in data.get("features", []):
        value = next(iter(feature["value"].values()))
        out[feature["name"]] = value["values"] if isinstance(value, dict) else value
    return out


def ratings_for(uid: str) -> list[tuple[str, float]]:
    # Same ratings table Phase 1 loaded. This is the only extra read, and
    # only for users the feature store does not know.
    job = bq.query(
        f"SELECT CAST(movieId AS STRING) AS movie_id, rating "
        f"FROM `{PROJECT}.movielens.ratings` "
        f"WHERE CAST(userId AS STRING) = @uid",
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("uid", "STRING", uid)]
        ),
    )
    return [(row.movie_id, float(row.rating)) for row in job]


def genre_match(top_genre, genres) -> float:
    # features.sql uses STRPOS. An empty genre must not match every movie,
    # and "" in "Action" is true in Python.
    if not top_genre:
        return 0.0
    return float(str(top_genre) in str(genres))


def score_candidates(user: dict, candidates, k: int):
    # Same six-feature order the ranker was trained on.
    # A stand-in user uses this function too. Only the vector differs.
    rows = []
    kept = []
    for candidate in candidates:
        movie = MOVIES.get(candidate.id)
        if movie is None:
            continue
        rows.append(
            [
                float(user["avg_rating"]),
                float(user["n_ratings"]),
                float(movie["avg_rating"]),
                float(movie["n_ratings"]),
                float(candidate.distance),
                genre_match(user.get("top_genre"), movie["genres"]),
            ]
        )
        kept.append(candidate)
    pred = ranker.predict(instances=rows)
    top = sorted(zip(kept, pred.predictions), key=lambda pair: -pair[1])[:k]
    items = [
        {
            "movie_id": candidate.id,
            "title": MOVIES[candidate.id]["title"],
            "score": round(score, 2),
        }
        for candidate, score in top
    ]
    return items, pred.deployed_model_id


@app.get("/recommend/{user_id}")
def recommend(user_id: str, k: int = 10):
    t0 = time.perf_counter()
    user = read_user(user_id)
    source = "feature_store"
    if "embedding" not in user:
        # Ratings already in BigQuery. Nothing is written to the feature store.
        # The next export replaces this stand-in with a real user vector.
        user = stand_in(ratings_for(user_id), MOVIES, EMBEDDINGS) or {}
        source = "ratings" if user else "popular"
    t1 = time.perf_counter()
    if source == "popular":
        # No vector, so skip Vector Search and the ranker.
        items, variant = popular_items(MOVIES, k), "popular"
        t2 = t3 = t1
    else:
        candidates = index_ep.find_neighbors(
            deployed_index_id="movies_v1",
            queries=[list(user["embedding"])],
            num_neighbors=50,
        )[0]
        t2 = time.perf_counter()
        items, variant = score_candidates(user, candidates, k)
        t3 = time.perf_counter()
    ms = lambda a, b: round((b - a) * 1000, 1)
    return {
        "user_id": user_id,
        "variant": variant,
        "user_source": source,
        "items": items,
        "latency_ms": {
            "features": ms(t0, t1),
            "retrieval": ms(t1, t2),
            "ranking": ms(t2, t3),
            "total": ms(t0, t3),
        },
    }
