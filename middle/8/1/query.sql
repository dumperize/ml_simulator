SELECT 
    product_name, monday, max_price, y,
    y_lag_1, y_lag_2, y_lag_3, y_lag_4, y_lag_5, y_lag_6,
    y_avg_3, y_max_3, y_min_3,
    y_avg_6, y_max_6, y_min_6,
    y_all_lag_1, y_all_lag_2, y_all_lag_3, y_all_lag_4, y_all_lag_5, y_all_lag_6,
    
    SUM(y_avg_3) OVER (PARTITION BY monday) as y_all_avg_3, 
    GREATEST(
        y_all_lag_1, 
        y_all_lag_2,
        y_all_lag_3
    ) AS y_all_max_3,
    LEAST(
        y_all_lag_1, 
        y_all_lag_2,
        y_all_lag_3
    ) AS y_all_min_3,
    
    SUM(y_avg_6) OVER (PARTITION BY monday) as y_all_avg_6, 
    GREATEST(
        y_all_lag_1, 
        y_all_lag_2,
        y_all_lag_3,
        y_all_lag_4, 
        y_all_lag_5,
        y_all_lag_6
    ) AS y_all_max_6,
    LEAST(
        y_all_lag_1, 
        y_all_lag_2,
        y_all_lag_3,
        y_all_lag_4, 
        y_all_lag_5,
        y_all_lag_6
    ) AS y_all_min_6
FROM (
    SELECT 
        *,
        y_sum_3 / 3 as y_avg_3, y_max_3, IF (y_count_3 < 3, 0 , y_min_3) AS y_min_3,
        y_sum_6 / 6 as y_avg_6, y_max_6, IF (y_count_6 < 6, 0 , y_min_6) AS y_min_6,
        
        SUM(y_lag_1) OVER (ORDER BY monday RANGE BETWEEN 1 PRECEDING AND CURRENT ROW) AS y_all_lag_1,
        SUM(y_lag_2) OVER (ORDER BY monday RANGE BETWEEN 2 PRECEDING AND CURRENT ROW) AS y_all_lag_2,
        SUM(y_lag_3) OVER (ORDER BY monday RANGE BETWEEN 3 PRECEDING AND CURRENT ROW) AS y_all_lag_3,
        SUM(y_lag_4) OVER (ORDER BY monday RANGE BETWEEN 4 PRECEDING AND CURRENT ROW) AS y_all_lag_4,
        SUM(y_lag_5) OVER (ORDER BY monday RANGE BETWEEN 5 PRECEDING AND CURRENT ROW) AS y_all_lag_5,
        SUM(y_lag_6) OVER (ORDER BY monday RANGE BETWEEN 6 PRECEDING AND CURRENT ROW) AS y_all_lag_6
    FROM (
        SELECT
            product_name,
            monday,
            max_price,
            y,
            SUM(y) OVER (PARTITION BY product_name ORDER BY monday ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING) AS y_lag_1,
            SUM(y) OVER (PARTITION BY product_name ORDER BY monday ROWS BETWEEN 2 PRECEDING AND 2 PRECEDING) AS y_lag_2,
            SUM(y) OVER (PARTITION BY product_name ORDER BY monday ROWS BETWEEN 3 PRECEDING AND 3 PRECEDING) AS y_lag_3,
            SUM(y) OVER (PARTITION BY product_name ORDER BY monday ROWS BETWEEN 4 PRECEDING AND 4 PRECEDING) AS y_lag_4,
            SUM(y) OVER (PARTITION BY product_name ORDER BY monday ROWS BETWEEN 5 PRECEDING AND 5 PRECEDING) AS y_lag_5,
            SUM(y) OVER (PARTITION BY product_name ORDER BY monday ROWS BETWEEN 6 PRECEDING AND 6 PRECEDING) AS y_lag_6,
            
            SUM(y) OVER (PARTITION BY product_name ORDER BY monday ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS y_sum_3,
            MAX(y) OVER (PARTITION BY product_name ORDER BY monday ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS y_max_3,
            MIN(y) OVER (PARTITION BY product_name ORDER BY monday ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS y_min_3,
            COUNT(y) OVER (PARTITION BY product_name ORDER BY monday ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS y_count_3,
            
            SUM(y) OVER (PARTITION BY product_name ORDER BY monday ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING) AS y_sum_6,
            MAX(y) OVER (PARTITION BY product_name ORDER BY monday ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING) AS y_max_6,
            MIN(y) OVER (PARTITION BY product_name ORDER BY monday ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING) AS y_min_6,
            COUNT(y) OVER (PARTITION BY product_name ORDER BY monday ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING) AS y_count_6,
            
            SUM(y) OVER (PARTITION BY monday) AS y_all,
            COUNT(y) OVER (PARTITION BY monday) AS y_count
        FROM (
            SELECT 
                date_trunc('week', dt::timestamp) AS monday,
                product_name,
                MAX(price) AS max_price,
                COUNT(product_name) AS y
            FROM default.data_sales_train
            GROUP BY product_name, monday
        )
    )
)
ORDER BY product_name, monday
