SELECT 
    brand,
    COUNT(sku_type) AS count_sku
FROM sku_dict_another_one
WHERE brand IS NOT NULL
GROUP BY brand
ORDER BY count_sku DESC
LIMIT 10