-- This script demonstrates a production-ready approach to data preparation.
-- It transforms raw Olist data into a clean, daily summary table inside the database,
-- which is more efficient for large-scale datasets than processing in Python.

-- Step 1: CTE to join and clean core order and customer data.
WITH cleaned_orders AS (
    SELECT
        t1.order_id,
        t1.order_purchase_timestamp,
        t3.customer_unique_id AS customer_id
    FROM
        olist_orders_dataset AS t1
    JOIN
        olist_customers_dataset AS t3 ON t1.customer_id = t3.customer_id
    WHERE
        t1.order_status = 'delivered'
),

-- Step 2: CTE to aggregate payment data for each order.
order_summary AS (
    SELECT
        order_id,
        SUM(payment_value) AS total_order_value
    FROM
        olist_order_payments_dataset
    GROUP BY
        order_id
    HAVING
        SUM(payment_value) > 0
)

-- Step 3: Create the final, daily summary table.
-- This is the table that Python would ideally read from.
DROP TABLE IF EXISTS daily_customer_summary;
CREATE TABLE daily_customer_summary AS
SELECT
    c.customer_id,
    CAST(c.order_purchase_timestamp AS DATE) AS purchase_date,
    SUM(o.total_order_value) AS daily_spend
FROM
    cleaned_orders AS c
JOIN
    order_summary AS o ON c.order_id = o.order_id
GROUP BY
    c.customer_id,
    purchase_date;

-- Final check to see the newly created table
SELECT * FROM daily_customer_summary LIMIT 10;
