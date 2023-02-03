SELECT 
    vendor,
    COUNT(DISTINCT brand) AS brand
FROM sku_dict_another_one
GROUP BY vendor
ORDER BY brand DESC
LIMIT 10