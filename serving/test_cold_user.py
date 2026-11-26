# Local checks for the unknown-user stand-in. No GCP calls.
from serving.cold_user import popular_items, stand_in

MOVIES = {
    "1": {"title": "A", "genres": "Action|Sci-Fi", "n_ratings": 10, "avg_rating": 4.5},
    "2": {"title": "B", "genres": "Comedy", "n_ratings": 3, "avg_rating": 3.0},
    "3": {"title": "C", "genres": "Action", "n_ratings": 100, "avg_rating": 4.0},
    "4": {"title": "Empty", "genres": "Drama", "n_ratings": 0, "avg_rating": 0},
}
EMB = {"1": [3.0, 0.0], "2": [0.0, 4.0]}


def test_likes_set_the_vector_and_the_stats():
    user = stand_in([("1", 5.0), ("2", 3.0)], MOVIES, EMB)
    assert user is not None
    assert user["n_ratings"] == 2
    assert user["avg_rating"] == 4.0
    # Only movie 1 is a like. Action and Sci-Fi tie, so the name decides.
    assert user["top_genre"] == "Action"
    assert user["embedding"] == [1.0, 0.0]


def test_no_likes_still_uses_rated_movies():
    user = stand_in([("2", 2.0)], MOVIES, EMB)
    assert user is not None
    assert user["top_genre"] is None
    assert user["embedding"] == [0.0, 1.0]


def test_like_without_a_vector_falls_back_to_other_ratings():
    # Movie 9 was liked but is not in the index. Movie 2 was only rated 2.
    user = stand_in([("9", 5.0), ("2", 2.0)], MOVIES, EMB)
    assert user is not None
    assert user["embedding"] == [0.0, 1.0]
    assert user["top_genre"] is None


def test_no_ratings_has_no_stand_in():
    assert stand_in([], MOVIES, EMB) is None


def test_popular_skips_unrated_movies():
    items = popular_items(MOVIES, 2)
    assert [item["movie_id"] for item in items] == ["3", "1"]
    assert items[0]["score"] == 4.0
