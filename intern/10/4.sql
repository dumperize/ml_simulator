SELECT 
    vendor,
    COUNT( sku_type) AS sku
FROM sku_dict_another_one
WHERE vendor IS NOT NULL
GROUP BY vendor
ORDER BY sku DESC
LIMIT 10