select 
    DATE(timestamp) as day,
    user_id,
    COUNT(submit) as n_submits,
    COUNT(DISTINCT task_id) as n_tasks,
    SUM(is_solved) as n_solved
from default.churn_submits
group by user_id, day
order by user_id, day