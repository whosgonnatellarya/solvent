import polars as pl
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pl.read_csv("invoices_featured.csv")
feature_cols = ["historical_client_delay", "relative_amount", "is_weekend", "days_until_due", "avg_client_amount"]
target_col = "days_to_pay"

X = df.select(feature_cols).to_numpy()
y = df.select(target_col).to_numpy().ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train)
dtrain.set_float_info("label_lower_bound", y_train)
dtrain.set_float_info("label_upper_bound", y_train)

params = {
    "objective": "survival:aft",
    "eval_metric": "aft-nloglik",
    "tree_method": "hist",
    "learning_rate": 0.05,
}

model = xgb.train(params, dtrain, num_boost_round=100)

dtest = xgb.DMatrix(X_test)
predictions = model.predict(dtest)
print(predictions)

mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae:.2f} days")

model.save_model("model.json")
print("model saved!")

feature_names = ["historical_client_delay", "relative_amount", "is_weekend", "days_until_due", "avg_client_amount"]
importance = model.get_score(importance_type="gain")
named_importance = {feature_names[int(k[1:])]: v for k, v in importance.items()}
print("Feature importances:", named_importance)