"""Part D (2/3) - refit Part C's final model, save it with its metadata, prove the reload works."""
import joblib
import numpy as np
import pandas as pd

from src.shipment_prep import (DATA, FEATURES, MODELS, OUT, clean_shipments, make_final_pipeline,
                               metric_table, save_csv, three_way_split)


def main():
    config = pd.read_csv(OUT / "part_c_final_config.csv").iloc[0]
    max_depth = None if config["max_depth"] == "unlimited" else int(config["max_depth"])
    pipeline = make_final_pipeline(int(config["n_estimators"]), max_depth, int(config["min_samples_leaf"]))

    table = pd.read_csv(DATA / "modelling_table.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = three_way_split(table[FEATURES], table["is_delayed"], stratify=True)
    pipeline.fit(X_train, y_train)

    MODELS.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODELS / "delay_model.pkl")
    threshold = float(config["threshold"])
    save_csv(pd.DataFrame({"threshold": [threshold]}), MODELS / "delay_threshold.csv")
    save_csv(pd.DataFrame({"feature": FEATURES}), MODELS / "feature_order.csv")
    _, medians = clean_shipments(pd.read_csv(DATA / "raw_shipments.csv"))
    save_csv(pd.DataFrame({"column": list(medians), "median": list(medians.values())}), MODELS / "impute_values.csv")

    # Reload and compare probabilities on the 180 test rows.
    reloaded = joblib.load(MODELS / "delay_model.pkl")
    diff = np.abs(reloaded.predict_proba(X_test)[:, 1] - pipeline.predict_proba(X_test)[:, 1]).max()
    save_csv(metric_table([("test_rows_scored", len(X_test)), ("reload_max_abs_probability_diff", diff),
                           ("threshold", threshold)]), OUT / "part_d_reload_check.csv")
    print("Packaged model; reload diff", diff)


if __name__ == "__main__":
    main()
