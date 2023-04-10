SELECT 
    age,
    income,
    dependents,
    has_property,
    has_car,
    credit_score,
    job_tenure,
    has_education,
    has_education,
    loan_amount,
    date_diff(day, loan_start, loan_deadline) AS loan_period,
    if(date_diff(day, loan_deadline, loan_payed) > 0, date_diff(day, loan_deadline, loan_payed), 0) AS delay_days
FROM default.loan_delay_days
