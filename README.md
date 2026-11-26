# Vertex movie recommender

Two-tower retrieval plus a gradient-boosting ranker on MovieLens latest-small. Phases 1–7 are in this repo. After a warmup, a Cloud Run load test of users 1–200 measured **p50 494 ms** and **p95 591 ms**. The hourly resources (Vector Search replica, ranker endpoint, Bigtable feature store, Cloud Run service) were removed after that test. The bucket, BigQuery dataset, index object, and model artifact can still exist and are storage, not a running replica.

`config.py` is gitignored. Create it before any phase:

```python
PROJECT = "your-gcp-project"
REGION = "us-central1"
BUCKET = f"gs://{PROJECT}-data"
BQ_DATASET = "movielens"
```

## What a new machine needs

1. `gcloud` logged in, with a configuration whose project is `PROJECT` and whose region is `us-central1`.
2. Application Default Credentials: `gcloud auth application-default login`.
3. A bucket at `gs://PROJECT-data` and a BigQuery dataset `movielens` in that project.
4. A trainer service account `p2-trainer@PROJECT.iam.gserviceaccount.com` with `roles/aiplatform.user`, `roles/bigquery.jobUser`, `roles/bigquery.dataEditor`, and `roles/storage.objectAdmin`.
5. Python 3.10+ and a venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install google-cloud-aiplatform google-cloud-bigquery google-cloud-storage \
  pandas db-dtypes setuptools requests fastapi uvicorn
```

`setuptools` is required. Vertex packages the training script with it.

Run every command below from the repo root, with the venv active and the gcloud configuration pointed at the project.

## Phase 1 — load MovieLens

```bash
python -m data.load_movielens
```

Downloads ml-latest-small, writes `movielens.ratings` and `movielens.movies`, and uploads `gs://PROJECT-data/raw/ratings.csv`. Expect about 100836 ratings, 9742 movies, and 610 users.

## Phase 2 — offline feature tables

```bash
bq query --use_legacy_sql=false < data/features.sql
```

Builds `movielens.user_stats` and `movielens.movie_stats`. A like is a rating of 4 or higher. One user in this dataset never liked anything, so later embedding tables have 609 users.

Check The Matrix:

```bash
bq query --use_legacy_sql=false \
  "SELECT movie_id, title, genres FROM movielens.movie_stats WHERE title LIKE 'Matrix, The%'"
```

## Phase 3 — two-tower tune and export

```bash
python -m trainer.tune
python -m trainer.export
```

`tune` runs 8 trials, two at a time, on `n1-standard-8`, and writes `trainer/best_params.json`. The metric is recall@50. The recorded best trial is dim 64, lr about 0.014, recall@50 about 0.099.

`export` trains once more with those values and writes L2-normalized vectors to `gs://PROJECT-data/emb/movies/movies.json` and `emb/users/users.json`. Delete any 0-byte folder marker under `emb/movies/` before Phase 4, or Vector Search tries to parse it.

## Phase 4 — Vector Search

```bash
python -m serving.vector_index
```

Builds a Tree-AH index (64 dimensions, dot product, 50 neighbors) and deploys it as `movies_v1` with one replica. This bills per hour until you undeploy it. The script writes `serving/index_endpoint.json`.

## Phase 5 — embeddings in BigQuery and the online store

```bash
bq load --autodetect --replace --source_format=NEWLINE_DELIMITED_JSON \
  movielens.user_embeddings gs://PROJECT-data/emb/users/users.json

bq load --autodetect --replace --source_format=NEWLINE_DELIMITED_JSON \
  movielens.movie_embeddings gs://PROJECT-data/emb/movies/movies.json

bq query --use_legacy_sql=false '
CREATE OR REPLACE TABLE movielens.user_serving AS
SELECT s.user_id, s.n_ratings, s.avg_rating, s.top_genre, e.embedding,
       CURRENT_TIMESTAMP() AS feature_timestamp
FROM movielens.user_stats s
JOIN movielens.user_embeddings e ON s.user_id = CAST(e.user_id AS STRING)'

python -m serving.feature_store_setup
python -m serving.read_user
```

The online store is Bigtable (`recsys_online`, one node), not the old Optimized store. The Vertex service agent needs `roles/bigtable.user`, `roles/bigquery.dataViewer`, and `roles/bigquery.jobUser` or the sync fails. The store bills until it is deleted. `read_user` prints user 1.

## Phase 6 — ranker

```bash
bq query --use_legacy_sql=false < data/rank_train.sql
python -m trainer.launch_ranker
```

`rank_train` is one row per rating. The six features, in serving order, are user average, user count, movie average, movie count, embedding dot product, and genre match. Averages include the rating being predicted. That leakage is left in on purpose.

Training uses the sklearn 1.0 image and pip-installs `scikit-learn==1.5.2` and `numpy==1.23.5`. Serving uses `sklearn-cpu.1-5`. Older serving images cannot unpickle this model. The script writes `trainer/ranker_endpoint.json`. The deployed endpoint bills per hour. Recorded test RMSE was 0.721.

## Phase 7 — API and latency

Create `p2-api@PROJECT.iam.gserviceaccount.com` with `roles/aiplatform.user`, `roles/bigquery.jobUser`, and `roles/bigquery.dataViewer`. Wait a minute so the account exists before the bindings.

```bash
gcloud run deploy recsys-api --source serving/ --region us-central1 \
  --project "$PROJECT" \
  --no-allow-unauthenticated \
  --service-account "p2-api@${PROJECT}.iam.gserviceaccount.com" \
  --min-instances 0 \
  --set-env-vars "PROJECT=${PROJECT},INDEX_ENDPOINT=INDEX_RESOURCE,RANKER_ENDPOINT=RANKER_RESOURCE"
```

Use the resource names from `serving/index_endpoint.json` and `trainer/ranker_endpoint.json`.

```bash
URL=$(gcloud run services describe recsys-api --region us-central1 --format='value(status.url)')
python serving/loadtest.py "$URL"
```

The script warms users 1, 2, and 3, then times users 1–200. A missing embedding is a 404 and is skipped. Anything else aborts.

## Stop the hourly bill

```bash
gcloud ai index-endpoints undeploy-index INDEX_ID \
  --deployed-index-id=movies_v1 --region=us-central1 --quiet
gcloud ai endpoints undeploy-model ENDPOINT_ID \
  --deployed-model-id=DEPLOYED_MODEL_ID --region=us-central1 --quiet
gcloud run services delete recsys-api --region=us-central1 --quiet
```

Delete the feature store from Python with `FeatureOnlineStore("recsys_online").delete(force=True)`.

## Plan: a user with no embedding

The user tower is an id lookup. A person who was not in training has no row in `user_embeddings`, and `GET /recommend/{user_id}` returns 404. Calling the tower with a new id does not help: the lookup maps unknown ids to one shared out-of-vocabulary vector, which is the same for every new user.

What other systems do, and what fits this model:

- If the user tower takes features (genres, locale, onboarding picks), serve that tower online and search the movie index with its output. Google's two-tower reference architecture is built this way so a new query does not need a new vocabulary. This repo's tower does not take those features. It only takes the user id.
- If the tower is id-only, build a stand-in vector in the same space: the L2-normalized average of the movie embeddings the person already liked, or of movies in the genres they picked at signup. That is the usual centroid cold start. Downstream retrieval and ranking stay the same.
- If there is no history and no genre pick, do not invent a personal vector. Return popular movies from `movie_stats` (high `n_ratings`, high `avg_rating`) until the first ratings arrive. Red Hat's recommender quickstart does the same thing with popular items in the categories chosen at registration, and only switches to the two-tower table on the next batch job.

Serving change, when this is implemented:

1. On 404 from Feature Store, accept a small body: a few movie ids the user liked, or a list of genres.
2. Average those movies' vectors from `movie_embeddings` (or every indexed movie whose genres hit the picks). L2-normalize. That vector is the Vector Search query.
3. Fill ranker inputs from the same ratings: `u_avg`, `u_n`, and `top_genre`. `sim` is still the neighbor distance. `genre_match` uses `top_genre`.
4. Keep that vector in the request only. The next `trainer.export` writes a real id embedding once the user has ratings in `ratings`, and `user_serving` replaces the stand-in.

Do not retrain on every signup. One export a day is enough for this catalog.

## Plan: a movie that is not in the index

The movie tower is also an id lookup. A title added after export is in `movies` only if you load it, and it is invisible to Vector Search until its vector is in the index. About 18 movies in this dataset already have no embedding because nobody rated them.

What other systems do:

- A content tower (genres, year, title text) can embed a movie the moment it is created, with no ratings. Google's write-up of two-tower retrieval calls this out as the reason to keep the candidate tower on features, not only on ids. Training with the id sometimes dropped forces the content features to carry new items.
- The new vector is then upserted into the ANN index. A full rebuild is the simple daily path. An incremental update is the fresh path: compute the vector, append it, and let a periodic job rebuild. Vertex Matching Engine does this as an index update from a new JSONL file under the contents URI, not as an in-process insert.
- Until that vector exists, reserve a few candidate slots for new titles (same genre as the user) so they can collect ratings. Otherwise a new movie never gets retrieved, never gets ratings, and never earns an id embedding.

Serving change, when this is implemented:

1. On insert, write the row to `movies` and refresh `movie_stats`. The API loads `movie_stats` at startup, so a new revision (or a periodic reload) has to see the title.
2. For the gap before the next export, build a temporary vector as the L2-normalized mean of indexed movies that share its genres. Upsert that JSON line and update the Vector Search index, or, cheaper, append the movie to the 50 neighbors inside `/recommend` when `top_genre` is in its genres and let the ranker score it. Use the genre-mean vector as `sim` if it is not in the index.
3. On the daily job, rerun export so the id is in the vocabulary, write `emb/movies/movies.json`, and update the index from that file. Then rebuild `rank_train` only if you want the ranker to see the new title's ratings.
4. Later, add genre and year to the movie tower and keep the id embedding. New movies then get a real vector from metadata on the first insert, and the id part takes over after they have ratings.

## Sources

- Google Cloud, "Implement two-tower retrieval for large-scale candidate generation": https://docs.cloud.google.com/architecture/implement-two-tower-retrieval-large-scale-candidate-generation
- Red Hat, "Understanding the recommender system's two-tower model": https://developers.redhat.com/articles/2026/01/26/understanding-recommender-systems-two-tower-model
- Cold-start notes on content towers and user context: https://github.com/neurarch-ai/awesome-ml-system-design/blob/main/topics/18-cold-start-and-exploration.md
- Centroid fallback for a user with no embedding: https://aboutvectordatabase.com/learn/how-to-handle-cold-start-problems-with-embeddings/
