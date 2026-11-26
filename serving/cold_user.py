# Stand-in features for a user who is not in the feature store.
# The two-tower model only knows user ids from the last export, so a new
# id has no vector. We stay in the same embedding space by averaging the
# movie vectors they already liked, then L2-normalising, which is what the
# export does to every user vector. Stats match data/features.sql: every
# rating counts toward the average, and top_genre comes only from likes
# (rating >= 4).
LIKE = 4.0


def l2_mean(vectors: list[list[float]]) -> list[float] | None:
    # One user, a handful of movies. Plain Python keeps numpy out of the
    # Cloud Run image.
    if not vectors:
        return None
    total = [0.0] * len(vectors[0])
    for vector in vectors:
        for index, value in enumerate(vector):
            total[index] += float(value)
    norm = sum(value * value for value in total) ** 0.5
    if norm == 0:
        return None
    return [value / norm for value in total]


def top_genre(liked_ids: list[str], movies: dict) -> str | None:
    # Same idea as features.sql: the genre that shows up on the most likes.
    # SQL uses ANY_VALUE on a tie. We break ties by name so the result is stable.
    counts: dict[str, int] = {}
    for movie_id in liked_ids:
        movie = movies.get(str(movie_id))
        if movie is None:
            continue
        for genre in str(movie.get("genres") or "").split("|"):
            if not genre or genre == "(no genres listed)":
                continue
            counts[genre] = counts.get(genre, 0) + 1
    if not counts:
        return None
    return sorted(counts, key=lambda genre: (-counts[genre], genre))[0]


def stand_in(
    ratings: list[tuple[str, float]],
    movies: dict,
    embeddings: dict,
) -> dict | None:
    # No ratings means we have nothing to average. The caller shows popular movies.
    if not ratings:
        return None
    liked = [movie_id for movie_id, rating in ratings if rating >= LIKE]
    # Likes are what the two-tower trained on, so they win when we have
    # a vector for them. Otherwise use every rated movie that is in the index.
    liked_vectors = [embeddings[movie_id] for movie_id in liked if movie_id in embeddings]
    any_vectors = [
        embeddings[movie_id] for movie_id, _rating in ratings if movie_id in embeddings
    ]
    embedding = l2_mean(liked_vectors or any_vectors)
    if embedding is None:
        return None
    return {
        "avg_rating": sum(rating for _movie_id, rating in ratings) / len(ratings),
        "n_ratings": len(ratings),
        "top_genre": top_genre(liked, movies),
        "embedding": embedding,
    }


def popular_items(movies: dict, k: int) -> list[dict]:
    # Most-rated first, then highest average. Same shape as a ranked response.
    # score is the movie's own average, because there is no user to rank against.
    ranked = sorted(
        movies.items(),
        key=lambda item: (
            -float(item[1]["n_ratings"]),
            -float(item[1]["avg_rating"]),
            item[0],
        ),
    )
    items = []
    for movie_id, movie in ranked:
        if float(movie["n_ratings"]) <= 0:
            continue
        items.append(
            {
                "movie_id": movie_id,
                "title": movie["title"],
                "score": round(float(movie["avg_rating"]), 2),
            }
        )
        if len(items) == k:
            break
    return items
