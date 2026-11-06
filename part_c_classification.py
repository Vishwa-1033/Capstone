"""Part C - delay classifier, feature signal, tuning and the alert threshold."""
import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.shipment_prep import (DATA, FEATURES, NOISE_COLUMNS, OUT, make_final_pipeline, save_csv,
                               three_way_split)


def confusion_row(name, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"model": name, "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)}


def main():
    table = pd.read_csv(DATA / "modelling_table.csv")
    X, y = table[FEATURES], table["is_delayed"]
    X_train, X_val, X_test, y_train, y_val, y_test = three_way_split(X, y, stratify=True)
    print("delayed in train/val/test:", y_train.sum(), y_val.sum(), y_test.sum())

    # 2. Blind baseline + three classifiers on validation (default 0.5 rule).
    rows = [confusion_row("blind_all_on_time", y_val, [0] * len(y_val))]
    logit = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]).fit(X_train, y_train)
    rows.append(confusion_row("logistic_regression", y_val, logit.predict(X_val)))
    tree_f1 = {d: f1_score(y_val, DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
                           .fit(X_train, y_train).predict(X_val)) for d in [2, 3, 4, 5, 6, 8, 10]}
    best_d = min(tree_f1, key=lambda d: (-tree_f1[d], d))                  # highest F1, ties -> smaller d
    tree = DecisionTreeClassifier(criterion="gini", max_depth=best_d, random_state=42).fit(X_train, y_train)
    rows.append(confusion_row(f"decision_tree_depth_{best_d}", y_val, tree.predict(X_val)))
    forest = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    rows.append(confusion_row("random_forest_default", y_val, forest.predict(X_val)))
    save_csv(pd.DataFrame(rows), OUT / "part_c_models.csv")

    # 3. Which columns carry signal? Forest importance + ANOVA F-score, training rows only.
    selector_all = SelectKBest(f_classif, k="all").fit(X_train, y_train)
    scores = pd.DataFrame({"feature": FEATURES, "f_score": selector_all.scores_,
                           "rf_importance": forest.feature_importances_})
    scores["f_rank"] = scores["f_score"].rank(ascending=False, method="first").astype(int)
    scores["is_noise"] = scores["feature"].isin(NOISE_COLUMNS)
    save_csv(scores.sort_values("f_rank")[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]],
             OUT / "part_c_feature_scores.csv")
    k_rows = []
    for k in [5, 7, 10, 12, 17]:
        pipe = Pipeline([("selector", SelectKBest(f_classif, k=k)),
                         ("model", RandomForestClassifier(n_estimators=100, random_state=42))]).fit(X_train, y_train)
        k_rows.append({"k": k, "val_f1": f1_score(y_val, pipe.predict(X_val))})
    save_csv(pd.DataFrame(k_rows), OUT / "part_c_selectk.csv")

    # 4. Tuning on the 7 selected columns (selector fitted on training rows only). Test rows untouched.
    selector = SelectKBest(f_classif, k=7).fit(X_train, y_train)
    X_train7, X_val7 = selector.transform(X_train), selector.transform(X_val)
    grid = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}
    searches = {
        "grid_search": GridSearchCV(RandomForestClassifier(random_state=42), grid, cv=5),
        "randomized_search": RandomizedSearchCV(RandomForestClassifier(random_state=42), grid, n_iter=9,
                                                cv=5, random_state=42),
    }
    tuning, chosen_params = [], {}
    for name, search in searches.items():
        search.fit(X_train7, y_train)
        candidates = len(search.cv_results_["params"])
        tuning.append({"search": name, "candidates": candidates, "cv_folds": 5, "total_fits": candidates * 5,
                       "best_params": json.dumps(search.best_params_, sort_keys=True),
                       "best_cv_accuracy": search.best_score_,
                       "val_f1": f1_score(y_val, search.best_estimator_.predict(X_val7))})
        chosen_params[name] = dict(search.best_params_)
    default7 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train7, y_train)
    default_params = {"max_depth": None, "min_samples_leaf": 1, "n_estimators": 100}
    tuning.append({"search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0, "total_fits": 1,
                   "best_params": json.dumps(default_params, sort_keys=True), "best_cv_accuracy": float("nan"),
                   "val_f1": f1_score(y_val, default7.predict(X_val7))})
    chosen_params["default_forest_same_7_features"] = default_params
    save_csv(pd.DataFrame(tuning), OUT / "part_c_tuning.csv")

    # 5. Final configuration: highest val_f1; ties -> default, grid, randomized.
    priority = ["default_forest_same_7_features", "grid_search", "randomized_search"]
    chosen = max(tuning, key=lambda r: (r["val_f1"], -priority.index(r["search"])))["search"]
    params = chosen_params[chosen]
    final = make_final_pipeline(params["n_estimators"], params["max_depth"], params["min_samples_leaf"])
    final.fit(X_train, y_train)

    # 6. Alert threshold: highest t whose validation recall >= 0.80 (else 0.1).
    val_proba = final.predict_proba(X_val)[:, 1]
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    t_rows = []
    for t in thresholds:
        row = confusion_row("", y_val, (val_proba >= t).astype(int))
        row.pop("model")
        t_rows.append({"threshold": t, **row})
    t_df = pd.DataFrame(t_rows)
    ok = t_df[t_df["recall"] >= 0.80]
    threshold = float(ok["threshold"].max()) if len(ok) else 0.1
    save_csv(t_df, OUT / "part_c_thresholds.csv")

    # 7. One test evaluation at the chosen threshold.
    test_row = confusion_row("", y_test, (final.predict_proba(X_test)[:, 1] >= threshold).astype(int))
    test_row.pop("model")
    save_csv(pd.DataFrame([{"threshold": threshold, **test_row}]), OUT / "part_c_final_test.csv")
    save_csv(pd.DataFrame([{"selector_k": 7, "n_estimators": params["n_estimators"],
                            "max_depth": "unlimited" if params["max_depth"] is None else params["max_depth"],
                            "min_samples_leaf": params["min_samples_leaf"], "threshold": threshold,
                            "chosen_from": chosen}]), OUT / "part_c_final_config.csv")
    print("final config:", chosen, params, "threshold", threshold)


if __name__ == "__main__":
    main()
