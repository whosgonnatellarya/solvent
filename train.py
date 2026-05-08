import polars as pl
import xgboost as xgb
df = pl.read_csv("invoices_featured.csv")
feature_cols = ["historical_client_delay", "relative_amount", "is_weekend", "days_until_due", "avg_client_amount"]
target_col = "days_to_pay"
X = df.select(feature_cols)
y = df.select(target_col)
X = X.to_numpy()
y = y.to_numpy().ravel()