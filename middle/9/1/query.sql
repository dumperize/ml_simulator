SELECT 
    user_id, 
    item_id,
    qty,
    ROUND(price,2) AS price
FROM (
    SELECT 
        user_id, 
        item_id,
        SUM(units) AS qty
    FROM default.karpov_express_orders 
    WHERE timestamp::date >= %(start_date)s and timestamp::date <= %(end_date)s
    GROUP BY user_id, item_id
) as t1
LEFT JOIN (
    SELECT 
        item_id,
        AVG(price) as price
    FROM default.karpov_express_orders 
    WHERE timestamp::date >= %(start_date)s and timestamp::date <= %(end_date)s
    GROUP BY item_id
) as t2 ON t1.item_id = t2.item_id
ORDER BY user_id, item_id