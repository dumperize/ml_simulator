SELECT 
    sku,
    dates,
    AVG(price) AS price,
    COUNT(price) AS qty
FROM transactions
GROUP BY dates, sku
ORDER BY sku