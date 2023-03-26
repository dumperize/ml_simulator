SELECT 
    date_trunc('week', date)::date AS weeks,
    SUM(amount) AS sum_receipt
FROM new_payments
WHERE status = 'Confirmed'
GROUP BY weeks
ORDER BY weeks