-- Query 1: SELECT + WHERE
SELECT title, price_gbp, rating
FROM books
WHERE rating >= 4;

-- Query 2: ORDER BY + LIMIT
SELECT title, price_inr
FROM books
ORDER BY price_inr DESC
LIMIT 10;

-- Query 3: DISTINCT
SELECT DISTINCT category_name
FROM categories;

-- Query 4: IN + JOIN
SELECT b.title, c.category_name, b.rating
FROM books b
INNER JOIN categories c ON b.category_id = c.category_id
WHERE c.category_name IN ('Travel', 'Mystery')
ORDER BY b.rating DESC;

-- Query 5: BETWEEN
SELECT title, price_gbp
FROM books
WHERE price_gbp BETWEEN 20 AND 40
ORDER BY price_gbp;

-- Query 6: JOIN
SELECT b.title, b.rating, b.price_gbp, c.category_name
FROM books b
INNER JOIN categories c ON b.category_id = c.category_id
ORDER BY b.rating DESC, c.category_name, b.title;
