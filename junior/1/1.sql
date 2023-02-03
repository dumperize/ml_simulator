SELECT 
    DATE_TRUNC('month', date)::date AS time,
    mode,
    COALESCE((COUNT(amount) FILTER (WHERE status = 'Confirmed')::float / COUNT(amount)) * 100, 0) as percents
FROM new_payments
WHERE mode != 'Не определено'
GROUP BY time, mode
ORDER BY time, mode
