"""Shared constants, cleaning and feature building for the Delay Intelligence capstone.

Everything that must behave identically in training and in prediction lives here, so
part_a..part_d, package_model.py and predict_delay.py all import it (one copy of the logic).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT, MODELS = ROOT / "data", ROOT / "outputs", ROOT / "models"

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

# Weights above this are scanner-fault sentinels (the network accepts parcels up to 40 kg).
MAX_PLAUSIBLE_WEIGHT_KG = 100
MEDIAN_KEYS = ["distance_km", "parcel_weight_kg", "declared_value_inr"]


def clean_shipments(raw, medians=None):
    """Clean a raw shipments export. Returns (clean_df, medians).

    medians=None  -> compute the three medians here (historical table);
    medians=dict  -> reuse them (new parcels), so gaps are filled exactly as in training.
    """
    df = raw.copy()
    df = df.drop_duplicates().reset_index(drop=True)                          # 1. exact duplicates
    df["service_type"] = df["service_type"].str.strip().str.title()           # 2. three spellings remain
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")                 # 3. blank -> Unknown
    df["declared_value_inr"] = pd.to_numeric(                                 # 4. text -> number
        df["declared_value"].str.replace("\u20b9", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce")
    df = df.drop(columns=["declared_value"])
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])                 # 5. datetime
    df.loc[df["parcel_weight_kg"] > MAX_PLAUSIBLE_WEIGHT_KG, "parcel_weight_kg"] = np.nan  # 6. sentinel
    if medians is None:                                                       # 7. median fill
        medians = {k: float(df[k].median()) for k in MEDIAN_KEYS}
    for key in MEDIAN_KEYS:
        df[key] = df[key].fillna(medians[key])
    return df, {k: float(medians[k]) for k in MEDIAN_KEYS}


def build_features(clean, hubs):
    """Attach hub columns and the model inputs. Never touches delivery_hours."""
    hub_cols = ["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]
    df = clean.merge(hubs[hub_cols], on="hub_id", how="left", validate="many_to_one")
    assert len(df) == len(clean), "hub merge changed the row count"
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)
    df["log_declared_value"] = np.log(df["declared_value_inr"])
    return df


# ---------- helpers shared by the scripts ----------
def three_way_split(X, y, stratify=False):
    """70/15/15 split in two calls, random_state=42 both times (stratified for Part C/packaging)."""
    s1 = y if stratify else None
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=s1)
    s2 = y_tmp if stratify else None
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=s2)
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_final_pipeline(n_estimators=100, max_depth=None, min_samples_leaf=1):
    """Part C's final model: top-7 feature selector + random forest."""
    return Pipeline([
        ("selector", SelectKBest(f_classif, k=7)),
        ("model", RandomForestClassifier(random_state=42, n_estimators=n_estimators,
                                         max_depth=max_depth, min_samples_leaf=min_samples_leaf)),
    ])


def clean_value(v):
    """Ints stay ints, floats are rounded to 4 decimals (brief: every decimal to 4 places)."""
    if isinstance(v, (bool, np.bool_, str)) or v is None:
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    return round(float(v), 4)


def save_csv(df, path, **kwargs):
    """Write with index=False and floats rounded to 4 decimals."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(4)
        elif out[col].dtype == object:
            out[col] = pd.Series([clean_value(v) for v in out[col]], index=out.index, dtype=object)  # ints stay ints
    Path(path).parent.mkdir(exist_ok=True)
    out.to_csv(path, index=kwargs.pop("index", False), **kwargs)


def metric_table(pairs):
    """[(metric, value), ...] -> DataFrame with python-typed values (ints stay ints)."""
    return pd.DataFrame({"metric": [m for m, _ in pairs], "value": pd.Series([clean_value(v) for _, v in pairs], dtype=object)})
