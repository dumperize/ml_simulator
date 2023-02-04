SELECT day, wau 
FROM (
    SELECT 
        day,
        user_id,
        COUNT( DISTINCT user_id) OVER (ORDER BY day::timestamp RANGE BETWEEN 6*24*60*60 PRECEDING AND CURRENT ROW) AS wau
    FROM (
        SELECT 
            timestamp::date AS day,
            user_id
        FROM churn_submits
        GROUP BY day, user_id
    ) as t1
    ORDER BY day
) t2
GROUP BY day, wau
ORDER BY day