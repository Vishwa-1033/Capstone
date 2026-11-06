"""Part A - clean the raw export, build the modelling table, write the delay evidence."""
import numpy as np
import pandas as pd

from src.shipment_prep import (DATA, FEATURES, OUT, build_features, clean_shipments,
                               metric_table, save_csv)


def main():
    raw = pd.read_csv(DATA / "raw_shipments.csv")
    hubs = pd.read_csv(DATA / "hubs.csv")

    # "Before" counts: measured after removing duplicates, before any filling.
    dedup = raw.drop_duplicates()
    clean, medians = clean_shipments(raw)
    table = build_features(clean, hubs)
    table["is_delayed"] = (table["delivery_hours"] > table["promised_hours"]).astype(int)
    table.to_csv(DATA / "modelling_table.csv", index=False)

    hours = table["delivery_hours"]
    q1, q3 = hours.quantile(0.25), hours.quantile(0.75)
    lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)   # unrounded bounds
    checks = [
        ("raw_rows", len(raw)),
        ("duplicate_rows_removed", len(raw) - len(dedup)),
        ("rows_after_dedup", len(dedup)),
        ("missing_distance_km_before", dedup["distance_km"].isna().sum()),
        ("missing_parcel_weight_before", dedup["parcel_weight_kg"].isna().sum()),
        ("sentinel_weight_rows", (dedup["parcel_weight_kg"] > 100).sum()),
        ("missing_declared_value_before", dedup["declared_value"].isna().sum()),
        ("missing_payment_mode_before", dedup["payment_mode"].isna().sum()),
        ("service_type_spellings_before", dedup["service_type"].nunique()),
        ("service_type_spellings_after", table["service_type"].nunique()),
        ("median_distance_km", medians["distance_km"]),
        ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
        ("median_declared_value_inr", medians["declared_value_inr"]),
        ("rows_after_hub_merge", len(table)),
        ("nulls_in_features_and_target", table[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum()),
        ("delayed_count", table["is_delayed"].sum()),
        ("delay_rate", table["is_delayed"].mean()),
        ("declared_value_mean_inr", table["declared_value_inr"].mean()),
        ("declared_value_median_inr", table["declared_value_inr"].median()),
        ("log_declared_value_mean", np.log(table["declared_value_inr"]).mean()),
        ("log_declared_value_median", np.log(table["declared_value_inr"]).median()),
        ("hours_q1", q1), ("hours_q3", q3),
        ("hours_lower_bound", lower), ("hours_upper_bound", upper),
        ("hours_outlier_count", ((hours < lower) | (hours > upper)).sum()),
    ]
    save_csv(metric_table(checks), OUT / "part_a_checks.csv")

    # Delay evidence by service type, hub tier and distance band.
    table["distance_band"] = pd.cut(table["distance_km"], bins=[0, 200, 400, 600, 2000],
                                    labels=["0-200", "201-400", "401-600", "601+"], right=True)
    rows = []
    for dim in ["service_type", "hub_tier", "distance_band"]:
        g = table.groupby(dim, observed=True).agg(shipments=("is_delayed", "size"), delayed=("is_delayed", "sum"),
                                                  delay_rate=("is_delayed", "mean"), mean_hours=("delivery_hours", "mean"))
        for group, r in g.iterrows():
            rows.append({"dimension": dim, "group": str(group), "shipments": int(r["shipments"]),
                         "delayed": int(r["delayed"]), "delay_rate": r["delay_rate"], "mean_hours": r["mean_hours"]})
    save_csv(pd.DataFrame(rows), OUT / "part_a_delay_by_group.csv")

    # Correlation matrix of the numeric columns (row labels kept as first column).
    corr = table.drop(columns="distance_band").select_dtypes("number").corr().round(4)
    corr.index.name = "column"
    corr.to_csv(OUT / "part_a_correlations.csv")
    print(f"Wrote {len(table)} rows to data/modelling_table.csv and Part A outputs")


if __name__ == "__main__":
    main()
