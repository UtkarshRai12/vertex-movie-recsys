-- One training row per rating for the ranking model.
-- Features, in serving order: user average, user count, movie average,
-- movie count, two-tower dot product, genre match.
-- The averages include the rating being predicted. That leakage is deliberate
-- for this first version.
-- Run: bq query --use_legacy_sql=false < data/rank_train.sql
CREATE OR REPLACE TABLE movielens.rank_train AS
SELECT u.avg_rating AS u_avg, u.n_ratings AS u_n,
       m.avg_rating AS m_avg, m.n_ratings AS m_n,
       (SELECT SUM(a * b)
        FROM UNNEST(u.embedding) a WITH OFFSET i
        JOIN UNNEST(me.embedding) b WITH OFFSET j ON i = j) AS sim,
       IF(STRPOS(m.genres, u.top_genre) > 0, 1.0, 0.0) AS genre_match,
       r.rating
FROM movielens.ratings r
JOIN movielens.user_serving u ON u.user_id = CAST(r.userId AS STRING)
JOIN movielens.movie_stats m ON m.movie_id = CAST(r.movieId AS STRING)
JOIN movielens.movie_embeddings me ON CAST(me.id AS STRING) = m.movie_id;
