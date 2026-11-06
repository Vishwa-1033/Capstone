"""Part B - delivery-hours regression: split, scale without leakage, compare models."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.shipment_prep import DATA, FEATURES, OUT, metric_table, save_csv, three_way_split


def scores(model_name, split, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"model": model_name, "split": split, "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mse, "RMSE": float(np.sqrt(mse)), "R2": r2_score(y_true, y_pred)}


def candidates(tree_depth):
    return {
        "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "decision_tree": DecisionTreeRegressor(max_depth=tree_depth, random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    }


def main():
    table = pd.read_csv(DATA / "modelling_table.csv")
    X, y = table[FEATURES], table["delivery_hours"]
    X_train, X_val, X_test, y_train, y_val, y_test = three_way_split(X, y)

    # Scaler fitted on TRAINING rows only; test rows are only transformed with it.
    scaler = StandardScaler().fit(X_train)
    save_csv(metric_table([
        ("train_rows", len(X_train)), ("val_rows", len(X_val)), ("test_rows", len(X_test)),
        ("train_scaled_max_abs_mean", np.abs(scaler.transform(X_train).mean(axis=0)).max()),
        ("test_scaled_max_abs_mean", np.abs(scaler.transform(X_test).mean(axis=0)).max()),
    ]), OUT / "part_b_checks.csv")

    metrics = [scores("mean_baseline", "val", y_val, np.full(len(y_val), y_train.mean()))]

    # Tree-depth diagnostic: bias (shallow) versus variance (deep).
    depth_rows = []
    for d in [1, 2, 3, 4, 6, 8, 12, None]:
        tree = DecisionTreeRegressor(max_depth=d, random_state=42).fit(X_train, y_train)
        depth_rows.append({"max_depth": "unlimited" if d is None else d,
                           "train_rmse": float(np.sqrt(mean_squared_error(y_train, tree.predict(X_train)))),
                           "val_rmse": float(np.sqrt(mean_squared_error(y_val, tree.predict(X_val)))),
                           "_order": 10 ** 6 if d is None else d})
    depth_df = pd.DataFrame(depth_rows)
    best_row = depth_df.sort_values(["val_rmse", "_order"]).iloc[0]       # ties -> smaller depth
    best_depth = None if best_row["max_depth"] == "unlimited" else int(best_row["max_depth"])
    save_csv(depth_df.drop(columns="_order"), OUT / "part_b_tree_depth.csv")

    # Four candidates scored on validation.
    fitted = {}
    for name, model in candidates(best_depth).items():
        fitted[name] = model.fit(X_train, y_train)
        metrics.append(scores(name, "val", y_val, fitted[name].predict(X_val)))

    # Learning-rate experiment for gradient boosting.
    lr_rows = []
    for lr in [0.01, 0.1, 1.0]:
        gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42).fit(X_train, y_train)
        pred = gb.predict(X_val)
        lr_rows.append({"learning_rate": lr, "val_rmse": float(np.sqrt(mean_squared_error(y_val, pred))),
                        "val_r2": r2_score(y_val, pred)})
    save_csv(pd.DataFrame(lr_rows), OUT / "part_b_gb_learning_rate.csv")

    # Rank candidates, cross-validate the best two with ONE shared KFold object.
    ranked = sorted(fitted, key=lambda n: [m for m in metrics if m["model"] == n][0]["RMSE"])
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    X_dev, y_dev = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
    cv_rows = []
    for name in ranked[:2]:
        r2 = cross_val_score(candidates(best_depth)[name], X_dev, y_dev, cv=kfold)
        cv_rows.append({"model": name, "mean_r2": r2.mean(), "std_r2": r2.std()})
    save_csv(pd.DataFrame(cv_rows), OUT / "part_b_cv.csv")

    # Final model = lowest validation RMSE; scored on the test rows exactly once.
    winner = ranked[0]
    metrics.append(scores(winner, "test", y_test, fitted[winner].predict(X_test)))
    save_csv(pd.DataFrame(metrics), OUT / "part_b_metrics.csv")
    print("Validation ranking:", ranked, "| final model:", winner)


if __name__ == "__main__":
    main()
