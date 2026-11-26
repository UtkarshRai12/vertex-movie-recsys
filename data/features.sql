-- Offline feature tables for the recommender.
-- user_stats: one row per user (rating count, average, favourite genre).
-- movie_stats: one row per movie, including movies with no ratings.
-- user_id and movie_id are strings because Feature Store uses them as keys.
-- Run from the repo root: bq query --use_legacy_sql=false < data/features.sql

-- A "like" is a rating of 4 or higher. top_genre is the genre that user liked most.
CREATE OR REPLACE TABLE movielens.user_stats AS
WITH liked AS (
  SELECT r.userId, genre, COUNT(*) AS c
  FROM movielens.ratings r JOIN movielens.movies m USING (movieId),
       UNNEST(SPLIT(m.genres, '|')) AS genre
  WHERE r.rating >= 4
  GROUP BY 1, 2
),
top AS (
  SELECT userId, ARRAY_AGG(genre ORDER BY c DESC LIMIT 1)[OFFSET(0)] AS top_genre
  FROM liked GROUP BY userId
)
SELECT CAST(r.userId AS STRING) AS user_id,
       COUNT(*) AS n_ratings,
       AVG(r.rating) AS avg_rating,
       ANY_VALUE(top.top_genre) AS top_genre
FROM movielens.ratings r LEFT JOIN top USING (userId)
GROUP BY user_id;

-- LEFT JOIN keeps movies that nobody rated. Those get n_ratings = 0 and avg_rating = 0.
CREATE OR REPLACE TABLE movielens.movie_stats AS
SELECT CAST(m.movieId AS STRING) AS movie_id, m.title, m.genres,
       COUNT(r.rating) AS n_ratings,
       IFNULL(AVG(r.rating), 0) AS avg_rating
FROM movielens.movies m LEFT JOIN movielens.ratings r USING (movieId)
GROUP BY movie_id, m.title, m.genres;
