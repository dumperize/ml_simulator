SELECT 
    Id AS review_id,
    parseDateTime64BestEffort(toString(Time)) AS dt,
    Score AS rating,
    multiIf(Score = 1, 'negative', Score = 5 , 'positive', 'neutral') AS sentiment,
    Text AS review
FROM simulator.flyingfood_reviews
ORDER BY review_id