"""Part D (1/3) - hub profile, K-Means segments, separation evidence and tier mismatches."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.shipment_prep import DATA, OUT, save_csv

PROFILE_COLUMNS = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
IMPLIED_SEGMENT = {1: "Standard", 2: "Watch", 3: "Escalate"}


def across_over_within(Z, labels):
    """Mean distance between hubs in different clusters / mean distance within clusters (276 pairs)."""
    dist = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)
    i, j = np.triu_indices(len(Z), k=1)
    same = labels[i] == labels[j]
    return dist[i, j][~same].mean() / dist[i, j][same].mean()


def main():
    table = pd.read_csv(DATA / "modelling_table.csv")
    hubs = pd.read_csv(DATA / "hubs.csv")

    # 1. One row per hub, ascending hub_id (this order matters for K-Means).
    profile = (table.groupby("hub_id").agg(shipments=("is_delayed", "size"),
                                           mean_delivery_hours=("delivery_hours", "mean"),
                                           delay_rate=("is_delayed", "mean"),
                                           mean_distance_km=("distance_km", "mean"),
                                           mean_n_stops=("n_stops", "mean"))
               .reset_index().sort_values("hub_id").reset_index(drop=True))
    save_csv(profile, OUT / "hub_profile.csv")

    # 2. Scale first (distance-based clustering needs comparable scales), then K-Means.
    Z = StandardScaler().fit_transform(profile[PROFILE_COLUMNS])
    # 3. Separation evidence for K = 2..5.
    sep = []
    for k in [2, 3, 4, 5]:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Z)
        sizes = sorted(np.bincount(labels), reverse=True)
        sep.append({"K": k, "across_over_within": across_over_within(Z, labels),
                    "cluster_sizes": "-".join(str(int(s)) for s in sizes)})
    save_csv(pd.DataFrame(sep), OUT / "part_d_separation.csv")

    # 4. K = 3 (fixed by the business). Name clusters by mean delay rate, not by arbitrary id.
    labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(Z)
    profile["cluster"] = labels
    order = profile.groupby("cluster")["delay_rate"].mean().sort_values().index
    profile["segment"] = profile["cluster"].map(dict(zip(order, ["Standard", "Watch", "Escalate"])))

    segments = profile[["hub_id", "segment"]].merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id")
    save_csv(segments, OUT / "hub_segments.csv")
    seg_profile = (profile.groupby("segment").agg(hubs=("hub_id", "size"),
                                                  mean_delivery_hours=("mean_delivery_hours", "mean"),
                                                  delay_rate=("delay_rate", "mean"),
                                                  mean_distance_km=("mean_distance_km", "mean"),
                                                  mean_n_stops=("mean_n_stops", "mean"))
                   .reindex(["Standard", "Watch", "Escalate"]).reset_index())
    save_csv(seg_profile, OUT / "segment_profile.csv")
    mismatch = segments[segments["segment"] != segments["hub_tier"].map(IMPLIED_SEGMENT)]
    save_csv(mismatch, OUT / "tier_vs_segment_mismatch.csv")
    print(seg_profile.round(4).to_string(index=False))
    print(mismatch.to_string(index=False))


if __name__ == "__main__":
    main()
