"""Part D (3/3) - score the 48 new parcels and turn probabilities into actions."""
import joblib
import pandas as pd

from src.shipment_prep import DATA, MODELS, OUT, build_features, clean_shipments, save_csv


def main():
    raw = pd.read_csv(DATA / "new_shipments.csv")
    hubs = pd.read_csv(DATA / "hubs.csv")
    model = joblib.load(MODELS / "delay_model.pkl")
    threshold = float(pd.read_csv(MODELS / "delay_threshold.csv")["threshold"].iloc[0])
    feature_order = pd.read_csv(MODELS / "feature_order.csv")["feature"].tolist()
    impute = pd.read_csv(MODELS / "impute_values.csv")
    medians = dict(zip(impute["column"], impute["median"].astype(float)))
    segments = pd.read_csv(OUT / "hub_segments.csv")[["hub_id", "segment"]]

    # Same cleaning and features as training (shared functions), with the STORED medians.
    clean, _ = clean_shipments(raw, medians)
    features = build_features(clean, hubs)
    X = features[feature_order]
    assert X.isna().sum().sum() == 0, "missing values left in the model inputs"

    result = features[["shipment_id", "hub_id"]].copy()
    result["delay_probability"] = model.predict_proba(X)[:, 1].round(4)
    result["flagged"] = (result["delay_probability"] >= threshold).astype(int)
    result = result.merge(segments, on="hub_id", how="left", validate="many_to_one")

    def action(row):
        if row["flagged"] == 1 and row["segment"] == "Escalate":
            return "Escalate to hub manager"
        return "Proactive customer call" if row["flagged"] == 1 else "No action"

    result["action"] = result.apply(action, axis=1)
    result = result[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
    save_csv(result, OUT / "new_shipment_actions.csv")
    print(f"Scored {len(result)} parcels: {result['flagged'].sum()} flagged, "
          f"{(result['action'] == 'Escalate to hub manager').sum()} escalations")


if __name__ == "__main__":
    main()
