with confirmed_payments AS (
    SELECT 
        DATE_TRUNC('month', date)::date as time,
        *
    FROM new_payments
    WHERE status = 'Confirmed'
)

SELECT 
    time,
    arppu,
    aov
FROM (
    SELECT 
        time,
        AVG(ppu) AS arppu
    FROM (
        SELECT 
            time,
            SUM(amount) AS ppu
        FROM confirmed_payments
        GROUP BY email_id, time
    ) AS t
    GROUP BY time
) as t1
LEFT JOIN (
    SELECT 
        time,
        AVG(amount) AS aov
    FROM confirmed_payments
    GROUP BY time
) as t2 USING(time)
ORDER BY time
