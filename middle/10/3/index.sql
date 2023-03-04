SELECT
    day,
    user_id,
    days_offline AS days_offline,
    sum_submits_14d / 14 as avg_submits_14d,
    IF (sum_submits = 0, 0, sum_solved / sum_submits) AS success_rate_14d,
    acc_sum AS solved_total,
    (target_14d = 0)::int AS target_14d
FROM (
    SELECT 
        *,
        IF(last_day IS NULL, NULL, DATEDIFF(day, last_day, day)::float) AS days_offline,
        SUM(n_solved) OVER (PARTITION BY user_id ORDER BY day) as acc_sum,
        SUM(n_submits) OVER (PARTITION BY user_id ORDER BY day RANGE BETWEEN 13 PRECEDING AND CURRENT ROW) AS sum_submits_14d,
        SUM(n_solved) OVER (PARTITION BY user_id ORDER BY day RANGE BETWEEN 13 PRECEDING AND CURRENT ROW) AS sum_solved,
        SUM(n_submits) OVER (PARTITION BY user_id ORDER BY day RANGE BETWEEN 13 PRECEDING AND CURRENT ROW) AS sum_submits,
        SUM(n_submits) OVER (PARTITION BY user_id ORDER BY day RANGE BETWEEN 1 FOLLOWING AND 14 FOLLOWING) AS target_14d
    FROM (
        SELECT 
            *,
            LAST_VALUE(
                CASE
                    WHEN n_submits > 0
                    THEN day
                    ELSE NULL
                END
            ) OVER (PARTITION BY user_id ORDER BY user_id, day) AS last_day
        FROM (
            SELECT 
                carcas.day,
                carcas.user_id, 
                n_submits,
                n_tasks,
                n_solved
            FROM (
                SELECT * FROM (
                    SELECT DISTINCT DATE(timestamp) as day  FROM default.churn_submits
                ) as days
                CROSS JOIN (SELECT DISTINCT user_id as user_id FROM  default.churn_submits) as users
                ORDER BY user_id, day
            ) as carcas
            LEFT JOIN (
                SELECT 
                    DATE(timestamp) as day,
                    user_id,
                    COUNT(submit) as n_submits,
                    COUNT(DISTINCT task_id) as n_tasks,
                    SUM(is_solved) as n_solved
                FROM default.churn_submits
                GROUP by user_id, day
            ) as counter ON counter.user_id = carcas.user_id and counter.day = carcas.day
        )
    ) 
)
ORDER BY user_id, day