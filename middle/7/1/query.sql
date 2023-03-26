WITH sum_amount_t AS (
        SELECT 
            SUM(amount) as sum_amount
        FROM new_payments
        WHERE status = 'Confirmed' and (mode = 'MasterCard' or mode = 'МИР' or mode = 'Visa')
        GROUP BY email_id
        ORDER BY sum_amount
) 
SELECT
    CASE
        WHEN order_point = 1 THEN '0-20000'
        WHEN order_point = 2 THEN '20000-40000'
        WHEN order_point = 3 THEN '40000-60000'
        WHEN order_point = 4 THEN '60000-80000'
        WHEN order_point = 5 THEN '80000-100000'
        ELSE CONCAT('100000-', ROUND(max_sum_amount))
    END AS purchase_range,
    num_of_users
FROM (
    SELECT 
            COUNT(*) AS num_of_users,
            MAX(sum_amount) AS max_sum_amount,
            CASE
                WHEN sum_amount <= 20000 THEN 1
                WHEN sum_amount <= 40000 THEN 2
                WHEN sum_amount <= 60000 THEN 3
                WHEN sum_amount <= 80000 THEN 4
                WHEN sum_amount <= 100000 THEN 5
                ELSE 6
            END AS order_point
    FROM sum_amount_t
    GROUP BY order_point
) AS t
ORDER BY order_point