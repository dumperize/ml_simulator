
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
ORDER BY user_id, day