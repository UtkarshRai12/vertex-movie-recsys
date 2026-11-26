# Real-time recommender. One request does three timed stages:
# Feature Store read, Vector Search, then the ranker.
# The movie table is small and static, so it is loaded once at startup.
import os
import time

from fastapi import FastAPI, HTTPException
from google.cloud import aiplatform, bigquery
from vertexai.resources.preview.feature_store import FeatureView

PROJECT = os.environ["PROJECT"]
REGION = "us-central1"
aiplatform.init(project=PROJECT, location=REGION)
user_view = FeatureView("user_features", feature_online_store_id="recsys_online")
index_ep = aiplatform.MatchingEngineIndexEndpoint(os.environ["INDEX_ENDPOINT"])
ranker = aiplatform.Endpoint(os.environ["RANKER_ENDPOINT"])
# movie_id is a string, matching the ids Vector Search returns.
MOVIES = (
    bigquery.Client(project=PROJECT)
    .query(f"SELECT * FROM `{PROJECT}.movielens.movie_stats`")
    .to_dataframe()
    .set_index("movie_id")
    .to_dict("index")
)
app = FastAPI()


def read_user(uid: str) -> dict:
    # Each feature value is wrapped, for example {"double_value": 4.1}.
    # Embeddings are {"double_array_value": {"values": [...]}}.
    data = user_view.read(key=[uid]).to_dict()
    out = {}
    for feature in data.get("features", []):
        value = next(iter(feature["value"].values()))
        out[feature["name"]] = value["values"] if isinstance(value, dict) else value
    return out


@app.get("/recommend/{user_id}")
def recommend(user_id: str, k: int = 10):
    t0 = time.perf_counter()
    user = read_user(user_id)
    if "embedding" not in user:
        raise HTTPException(404, "unknown user")
    t1 = time.perf_counter()
    candidates = index_ep.find_neighbors(
        deployed_index_id="movies_v1",
        queries=[user["embedding"]],
        num_neighbors=50,
    )[0]
    t2 = time.perf_counter()
    # Same six-feature order the ranker was trained on.
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
                float(str(user.get("top_genre")) in movie["genres"]),
            ]
        )
        kept.append(candidate)
    pred = ranker.predict(instances=rows)
    t3 = time.perf_counter()
    top = sorted(zip(kept, pred.predictions), key=lambda pair: -pair[1])[:k]
    ms = lambda a, b: round((b - a) * 1000, 1)
    return {
        "user_id": user_id,
        "variant": pred.deployed_model_id,
        "items": [
            {
                "movie_id": candidate.id,
                "title": MOVIES[candidate.id]["title"],
                "score": round(score, 2),
            }
            for candidate, score in top
        ],
        "latency_ms": {
            "features": ms(t0, t1),
            "retrieval": ms(t1, t2),
            "ranking": ms(t2, t3),
            "total": ms(t0, t3),
        },
    }
