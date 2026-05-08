import polars as pl
from generate_data import df

df = df.with_columns(
    (pl.col("paid_date") - pl.col("issue_date")).dt.total_days().alias("days_to_pay")
)

client_delays = df.group_by("client_name").agg(
    pl.col("days_to_pay").mean().alias("historical_client_delay")
)
df = df.join(client_delays, on="client_name", how="left")

df = df.with_columns(
    pl.col("historical_client_delay").cast(pl.Float64)
)

client_amounts = df.group_by("client_name").agg(
    pl.col("amount").mean().alias("avg_client_amount")
)
df = df.join(client_amounts, on="client_name", how="left")

df = df.with_columns(
    (pl.col("amount") / pl.col("avg_client_amount")).alias("relative_amount")
)

df = df.with_columns(
    (pl.col("due_date") - pl.col("issue_date")).dt.total_days().alias("days_until_due")
)

df = df.with_columns(
    pl.when((pl.col("issue_date").dt.weekday() == 5) | (pl.col("issue_date").dt.weekday() == 6)).then(1).otherwise(0).alias("is_weekend")
)

print(df)
df.write_csv("invoices_featured.csv")