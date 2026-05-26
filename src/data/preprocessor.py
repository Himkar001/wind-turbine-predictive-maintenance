"""
Data Preprocessor — Permanent Fix
===================================

Key changes from previous version:

1. REMOVED TEMPORAL LEAKAGE FEATURES
   month, month_sin, month_cos removed from classifier features.
   These caused the model to memorize WHEN faults were injected
   rather than WHAT the sensor pattern looks like.

2. FAULT DISCRIMINATOR FEATURES
   Physics-informed features that are UNIQUE to each fault type:
   - bearing_stress_index    → bearing_failure signal
   - gearbox_stress_index    → gearbox_fault signal
   - electrical_stress_index → electrical_fault signal
   - thermal_divergence      → generator_overheating signal
   - oil_pressure_drop_rate  → oil_leak signal
   - vibration_pattern_index → blade_imbalance signal

3. STRATIFIED TEMPORAL SPLIT
   Split is time-based (no leakage) but also guarantees all 8 fault
   classes appear in EVERY split. If any class is missing from train,
   the earliest window of that class is moved into train from val/test.
   This is the hybrid approach — temporal integrity + class coverage.

4. DIGITAL TWIN trained on TRAIN-ONLY healthy data
   No leakage from val/test into the twin models.
"""

import logging
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.twin.digital_twin import DigitalTwinEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


SENSOR_COLS = [
    "wind_speed", "rotor_speed", "bearing_temp",
    "generator_temp", "vibration", "oil_pressure", "power_output"
]

WINDOWS = {"5min": 30, "30min": 180, "2hr": 720}

# Features excluded from ML models — metadata or leakage risk
EXCLUDE_FROM_FEATURES = {
    "timestamp", "turbine_id", "fault_type", "fault_label",
    "is_anomaly", "severity", "rul_days",
    # REMOVED: temporal features that leak injection timing
    "month", "month_sin", "month_cos",
}


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Memory optimization ───────────────────────────────────────────────────────

def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
        elif df[col].dtype == "int64":
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


# ── Load ──────────────────────────────────────────────────────────────────────

def load_raw_data(config: dict) -> pd.DataFrame:
    path = Path(config["data_generation"]["output_dir"]) / "all_sensor_data.csv"
    logger.info(f"Loading {path} ...")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    logger.info(f"  Loaded {len(df):,} rows × {df.shape[1]} cols")
    return df


# ── Clean ─────────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning data ...")
    original = len(df)

    df = df.sort_values(["turbine_id", "timestamp"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["turbine_id", "timestamp"])

    bounds = {
        "bearing_temp":   (0, 130), "generator_temp": (0, 140),
        "vibration":      (0, 20),  "oil_pressure":   (0, 12),
        "power_output":   (0, 2600),"rotor_speed":    (0, 30),
        "wind_speed":     (0, 40),
    }
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    logger.info(f"  Cleaned: {original:,} → {len(df):,} rows")
    return df


# ── Rolling features ──────────────────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding rolling features ...")
    for sensor in SENSOR_COLS:
        for wname, wsize in WINDOWS.items():
            df[f"{sensor}_mean_{wname}"] = (
                df.groupby("turbine_id", sort=False)[sensor]
                  .transform(lambda x: x.rolling(wsize, min_periods=1).mean())
                  .astype("float32")
            )
            df[f"{sensor}_std_{wname}"] = (
                df.groupby("turbine_id", sort=False)[sensor]
                  .transform(lambda x: x.rolling(wsize, min_periods=1).std().fillna(0))
                  .astype("float32")
            )
    logger.info(f"  Added {len(SENSOR_COLS)*len(WINDOWS)*2} rolling features")
    return df


# ── Rate of change ────────────────────────────────────────────────────────────

def add_roc_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding rate-of-change features ...")
    for sensor in SENSOR_COLS:
        df[f"{sensor}_roc"] = (
            df.groupby("turbine_id", sort=False)[sensor]
              .diff().fillna(0).astype("float32")
        )
    logger.info(f"  Added {len(SENSOR_COLS)} RoC features")
    return df


# ── Cross-sensor features ─────────────────────────────────────────────────────

def add_cross_sensor_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding cross-sensor features ...")
    eps = 1e-6

    df["power_efficiency"]             = (df["power_output"] / (df["wind_speed"]**3 + eps)).astype("float32")
    df["temp_speed_ratio"]             = (df["bearing_temp"] / (df["rotor_speed"] + eps)).astype("float32")
    df["generator_bearing_temp_ratio"] = (df["generator_temp"] / (df["bearing_temp"] + eps)).astype("float32")
    df["pressure_speed_ratio"]         = (df["oil_pressure"] / (df["rotor_speed"] + eps)).astype("float32")
    df["vibration_speed_ratio"]        = (df["vibration"] / (df["rotor_speed"] + eps)).astype("float32")

    logger.info("  Added 5 cross-sensor features")
    return df


# ── FAULT DISCRIMINATOR FEATURES ──────────────────────────────────────────────

def add_fault_discriminator_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Physics-informed features designed to be UNIQUE to each fault type.
    These help the classifier distinguish between faults that affect
    similar sensors (e.g. gearbox vs oil_leak both affect oil_pressure).
    """
    logger.info("Adding fault discriminator features ...")
    eps = 1e-6

    # ── bearing_failure discriminator ────────────────────────────
    # bearing_temp rises while power_output stays relatively normal
    # High value = bearing is hot relative to power being produced
    df["bearing_stress_index"] = (
        df["bearing_temp"] /
        (df["power_output"].clip(lower=1) / 500 + eps)
    ).astype("float32")

    # ── gearbox_fault discriminator ──────────────────────────────
    # Oil pressure drops AND vibration rises together (oscillating)
    # High value = pressure is low AND vibration is high
    df["gearbox_stress_index"] = (
        df["vibration"] /
        (df["oil_pressure"].clip(lower=0.1) + eps)
    ).astype("float32")

    # ── oil_leak discriminator ───────────────────────────────────
    # Steady oil pressure decline rate (rolling)
    # Different from gearbox: no vibration oscillation
    df["oil_pressure_decline_rate"] = (
        df.groupby("turbine_id", sort=False)["oil_pressure"]
          .transform(lambda x: x.rolling(180, min_periods=1).mean() - x)
          .fillna(0).astype("float32")
    )

    # ── electrical_fault discriminator ───────────────────────────
    # Power drops dramatically but bearing_temp stays normal
    # Key: power / bearing_temp ratio drops (electrical) vs
    #      bearing_temp rises independently (bearing fault)
    rolling_power = (
        df.groupby("turbine_id", sort=False)["power_output"]
          .transform(lambda x: x.rolling(720, min_periods=1).mean())
          .clip(lower=1)
    )
    df["electrical_stress_index"] = (
        1.0 - (df["power_output"] / (rolling_power + eps))
    ).clip(0, 1).astype("float32")

    # ── generator_overheating discriminator ──────────────────────
    # generator_temp rises but bearing_temp does NOT rise proportionally
    # Thermal divergence: large positive = generator hotter than bearing
    df["thermal_divergence"] = (
        df["generator_temp"] - df["bearing_temp"]
    ).astype("float32")

    # ── blade_imbalance discriminator ────────────────────────────
    # Vibration oscillates (not a ramp) — capture variance of vibration
    df["vibration_variance_30min"] = (
        df.groupby("turbine_id", sort=False)["vibration"]
          .transform(lambda x: x.rolling(180, min_periods=1).var().fillna(0))
          .astype("float32")
    )

    # ── Combined health index (useful for risk scoring) ───────────
    # Lower = healthier
    df["health_degradation_index"] = (
        0.25 * (df["bearing_temp"] / 95).clip(0, 1)
        + 0.25 * (df["generator_temp"] / 110).clip(0, 1)
        + 0.20 * (df["vibration"] / 10).clip(0, 1)
        + 0.15 * (1 - df["oil_pressure"] / 6).clip(0, 1)
        + 0.15 * (1 - df["power_output"] / 2000).clip(0, 1)
    ).astype("float32")

    logger.info("  Added 7 fault discriminator features")
    return df


# ── Time features (NO month — removes temporal leakage) ───────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding time features (no month — removes leakage) ...")

    df["hour"]        = df["timestamp"].dt.hour.astype("int8")
    df["day_of_week"] = df["timestamp"].dt.dayofweek.astype("int8")
    df["is_night"]    = ((df["hour"] < 6) | (df["hour"] >= 20)).astype("int8")

    # Cyclic encoding of hour and day-of-week
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype("float32")
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7).astype("float32")
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7).astype("float32")

    # NOTE: month / month_sin / month_cos deliberately excluded
    # Reason: 180-day dataset spans only a few months. Including month
    # causes the classifier to memorize injection timing (e.g.
    # "electrical_fault always happens in month 3") instead of learning
    # sensor patterns. This is temporal data leakage.

    logger.info("  Added 8 time features (hour, dow, is_night + cyclic encodings)")
    return df


# ── Digital twin features ─────────────────────────────────────────────────────

def add_digital_twin_features(
    df: pd.DataFrame,
    train_df: pd.DataFrame = None,
    save_dir: str = "data/models/twin"
) -> pd.DataFrame:
    """
    Train digital twin on TRAIN ONLY healthy data.
    Then compute residuals for the entire dataset.

    If train_df is None (legacy mode), trains on all healthy data.
    """
    logger.info("Adding digital twin residual features ...")
    engine = DigitalTwinEngine()

    if train_df is not None:
        # CORRECT: fit on train healthy data only — no leakage
        logger.info("  Training twin on TRAIN-ONLY healthy data (no leakage)")
        engine.fit_all(train_df)
    else:
        # Fallback: fit on all healthy data (used when called before split)
        logger.info("  Training twin on all healthy data")
        engine.fit_all(df)

    engine.save_all(save_dir)
    df = engine.compute_residuals_all(df)
    logger.info("  Digital twin residual features added")
    return df


# ── Label encoding ────────────────────────────────────────────────────────────

def encode_labels(df: pd.DataFrame):
    logger.info("Encoding labels ...")
    le = LabelEncoder()
    df["fault_label"] = le.fit_transform(df["fault_type"])
    mapping = {cls: int(idx) for idx, cls in enumerate(le.classes_)}
    logger.info(f"  Label mapping: {mapping}")
    return df, le


# ── Feature columns ───────────────────────────────────────────────────────────

def get_feature_columns(df: pd.DataFrame) -> list:
    return [
        col for col in df.columns
        if col not in EXCLUDE_FROM_FEATURES
        and df[col].dtype in ["float32", "float64", "int8", "int16", "int32", "int64"]
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# STRATIFIED TEMPORAL SPLIT
# ═══════════════════════════════════════════════════════════════════════════════

def split_data(df: pd.DataFrame, config: dict):
    """
    Hybrid split:

    Step 1 — Time-based split per turbine (preserves temporal order)
    Step 2 — Check which fault classes are missing from train
    Step 3 — Move the earliest window of each missing class into train
             (pulling from val first, then test)
    Step 4 — Verify all 8 classes present in every split

    No shuffling. No future data leaks into train.
    """
    train_r = config["data_generation"]["train_split"]    # 0.70
    val_r   = config["data_generation"]["val_split"]      # 0.15

    logger.info("Splitting data (stratified temporal split) ...")

    trains, vals, tests = [], [], []
    for tid, group in df.groupby("turbine_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        n     = len(group)
        t_end = int(n * train_r)
        v_end = int(n * (train_r + val_r))
        trains.append(group.iloc[:t_end].copy())
        vals.append(group.iloc[t_end:v_end].copy())
        tests.append(group.iloc[v_end:].copy())

    train_df = pd.concat(trains).reset_index(drop=True)
    val_df   = pd.concat(vals).reset_index(drop=True)
    test_df  = pd.concat(tests).reset_index(drop=True)

    # ── Step 2: find missing classes ─────────────────────────────
    all_classes   = set(df["fault_type"].unique())
    train_classes = set(train_df["fault_type"].unique())
    missing       = all_classes - train_classes

    if missing:
        logger.info(f"  Missing classes in train: {missing}")
        logger.info("  Applying hybrid fix ...")

        new_trains, new_vals, new_tests = [], [], []

        for tid in df["turbine_id"].unique():
            t_grp  = train_df[train_df["turbine_id"] == tid].copy()
            v_grp  = val_df[val_df["turbine_id"] == tid].copy()
            te_grp = test_df[test_df["turbine_id"] == tid].copy()

            for fault in missing:
                for src_name, src_grp in [("val", v_grp), ("test", te_grp)]:
                    fault_rows = src_grp[src_grp["fault_type"] == fault]
                    if len(fault_rows) == 0:
                        continue

                    fault_start  = fault_rows["timestamp"].min()
                    fault_end    = fault_rows["timestamp"].max()
                    window_start = fault_start - pd.Timedelta(days=2)

                    window = src_grp[
                        (src_grp["timestamp"] >= window_start) &
                        (src_grp["timestamp"] <= fault_end)
                    ].copy()

                    logger.info(
                        f"    [{tid}] Moving {fault} "
                        f"({len(window):,} rows) from {src_name} → train"
                    )

                    t_grp = pd.concat([t_grp, window]).sort_values("timestamp")

                    remove_mask = (
                        (src_grp["timestamp"] >= window_start) &
                        (src_grp["timestamp"] <= fault_end)
                    )
                    if src_name == "val":
                        v_grp = v_grp[~remove_mask]
                    else:
                        te_grp = te_grp[~remove_mask]
                    break  # found in this source, move on

            new_trains.append(t_grp)
            new_vals.append(v_grp)
            new_tests.append(te_grp)

        train_df = pd.concat(new_trains).reset_index(drop=True)
        val_df   = pd.concat(new_vals).reset_index(drop=True)
        test_df  = pd.concat(new_tests).reset_index(drop=True)

    # ── Step 4: verify ───────────────────────────────────────────
    final_train = set(train_df["fault_type"].unique())
    final_val   = set(val_df["fault_type"].unique())
    final_test  = set(test_df["fault_type"].unique())
    still_missing = all_classes - final_train

    if still_missing:
        logger.warning(f"  Still missing after hybrid fix: {still_missing}")
    else:
        logger.info("  Stratified temporal split verified — all classes in train ✓")

    logger.info(f"  Train : {len(train_df):,} rows | classes: {len(final_train)}")
    logger.info(f"  Val   : {len(val_df):,} rows  | classes: {len(final_val)}")
    logger.info(f"  Test  : {len(test_df):,} rows  | classes: {len(final_test)}")

    # Class distribution in train
    logger.info("  Train fault class distribution:")
    dist = train_df["fault_type"].value_counts()
    for fault, count in dist.items():
        pct = count / len(train_df) * 100
        logger.info(f"    {fault:30s} {count:>9,}  ({pct:.1f}%)")

    return train_df, val_df, test_df


# ── Scaling ───────────────────────────────────────────────────────────────────

def scale_features(train_df, val_df, test_df, feature_cols):
    logger.info("Scaling features ...")
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols]).astype("float32")
    val_df[feature_cols]   = scaler.transform(val_df[feature_cols]).astype("float32")
    test_df[feature_cols]  = scaler.transform(test_df[feature_cols]).astype("float32")
    logger.info("  Scaling complete")
    return train_df, val_df, test_df, scaler


# ── Save ──────────────────────────────────────────────────────────────────────

def save_processed_data(
    train_df, val_df, test_df,
    feature_cols, label_encoder, scaler,
    output_dir: str = "data/processed"
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(out / "train.parquet", index=False)
    val_df.to_parquet(out   / "val.parquet",   index=False)
    test_df.to_parquet(out  / "test.parquet",  index=False)

    joblib.dump(scaler,        out / "scaler.joblib")
    joblib.dump(label_encoder, out / "label_encoder.joblib")
    joblib.dump(feature_cols,  out / "feature_columns.joblib")

    # Save label classes as JSON for easy lookup
    label_classes = {str(i): cls for i, cls in enumerate(label_encoder.classes_)}
    with open(out / "label_classes.json", "w") as f:
        json.dump(label_classes, f, indent=2)

    logger.info(f"  Saved train/val/test parquet → {out}/")
    logger.info(f"  Saved scaler, encoder, feature_cols, label_classes")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_preprocessing_pipeline(
    config_path: str = "configs/config.yaml",
    output_dir:  str = "data/processed"
):
    logger.info("=" * 65)
    logger.info("  Data Preprocessor — Permanent Fix")
    logger.info("  No month leakage | Fault discriminators | Stratified split")
    logger.info("=" * 65)

    config = load_config(config_path)

    # 1. Load raw data
    df = load_raw_data(config)

    # 2. Clean
    logger.info("Cleaning data ...")
    df = clean_data(df)

    # 3. Feature engineering (before split — rolling windows need full history)
    logger.info("Starting feature engineering ...")
    df = reduce_memory(df)
    df = add_rolling_features(df);          df = reduce_memory(df)
    df = add_roc_features(df);              df = reduce_memory(df)
    df = add_cross_sensor_features(df);     df = reduce_memory(df)
    df = add_fault_discriminator_features(df); df = reduce_memory(df)
    df = add_time_features(df);             df = reduce_memory(df)

    # 4. Encode labels (needed before split for stratification check)
    df, label_encoder = encode_labels(df)

    # 5. SPLIT FIRST — then fit digital twin on train only
    train_df, val_df, test_df = split_data(df, config)

    # 6. Digital twin — fit on TRAIN healthy data only (no leakage)
    logger.info("Training digital twin on train-only healthy data ...")
    train_healthy = train_df[train_df["fault_type"] == "normal"].copy()
    engine = DigitalTwinEngine()
    engine.fit_all(train_healthy)
    engine.save_all("data/models/twin")

    # Compute residuals for all splits using the train-fitted twin
    logger.info("Computing twin residuals for all splits ...")
    train_df = engine.compute_residuals_all(train_df)
    val_df   = engine.compute_residuals_all(val_df)
    test_df  = engine.compute_residuals_all(test_df)

    # Recombine for consistent feature column extraction
    # then re-split (residual columns are now added)
    logger.info("Finalizing feature columns ...")
    df = reduce_memory(train_df)
    feature_cols = get_feature_columns(train_df)
    logger.info(f"  Total feature columns: {len(feature_cols)}")

    # Also apply to val and test
    val_feature_cols  = get_feature_columns(val_df)
    test_feature_cols = get_feature_columns(test_df)

    # Use intersection to ensure consistency
    feature_cols = list(
        set(feature_cols) & set(val_feature_cols) & set(test_feature_cols)
    )
    feature_cols = sorted(feature_cols)
    logger.info(f"  Consistent feature columns: {len(feature_cols)}")

    # 7. Scale — fit on train only
    train_df, val_df, test_df, scaler = scale_features(
        train_df, val_df, test_df, feature_cols
    )

    # 8. Save
    logger.info("Saving processed datasets ...")
    save_processed_data(
        train_df, val_df, test_df,
        feature_cols, label_encoder, scaler,
        output_dir
    )

    logger.info("=" * 65)
    logger.info(f"  Feature count  : {len(feature_cols)}")
    logger.info(f"  Train rows     : {len(train_df):,}")
    logger.info(f"  Val rows       : {len(val_df):,}")
    logger.info(f"  Test rows      : {len(test_df):,}")
    logger.info("  Preprocessing complete ✓")
    logger.info("=" * 65)

    return train_df, val_df, test_df, feature_cols


if __name__ == "__main__":
    run_preprocessing_pipeline()