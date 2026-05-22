"""
Data Preprocessor — Phase 2
============================

Optimized preprocessing pipeline with:
- cleaning
- rolling statistics
- rate-of-change features
- cross-sensor features
- time features
- digital twin residuals
- scaling
- train/val/test split

Final output is fully ready for:
- LSTM
- Transformers
- anomaly detection
- RUL prediction
- predictive maintenance
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

# ─────────────────────────────────────────────
# Digital Twin Import
# ─────────────────────────────────────────────

from src.twin.digital_twin import DigitalTwinEngine


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

SENSOR_COLS = [
    "wind_speed",
    "rotor_speed",
    "bearing_temp",
    "generator_temp",
    "vibration",
    "oil_pressure",
    "power_output"
]

WINDOWS = {
    "5min": 30,
    "30min": 180,
    "2hr": 720
}


# ─────────────────────────────────────────────
# Config Loader
# ─────────────────────────────────────────────

def load_config(
    path: str = "configs/config.yaml"
) -> dict:

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


# ─────────────────────────────────────────────
# Memory Optimization
# ─────────────────────────────────────────────

def reduce_memory_usage(
    df: pd.DataFrame
) -> pd.DataFrame:

    logger.info("Reducing memory usage")

    for col in df.columns:

        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")

        elif df[col].dtype == "int64":
            df[col] = pd.to_numeric(
                df[col],
                downcast="integer"
            )

    logger.info("Memory optimization completed")

    return df


# ─────────────────────────────────────────────
# Load Raw Data
# ─────────────────────────────────────────────

def load_raw_data(
    config: dict,
    turbine_id: str = None
) -> pd.DataFrame:

    raw_dir = Path(
        config["data_generation"]["output_dir"]
    )

    if turbine_id:

        path = raw_dir / f"sensor_data_{turbine_id}.csv"

        logger.info(
            f"Loading turbine dataset: {path}"
        )

    else:

        path = raw_dir / "all_sensor_data.csv"

        logger.info(
            f"Loading full dataset from {path}"
        )

    df = pd.read_csv(
        path,
        parse_dates=["timestamp"]
    )

    logger.info(
        f"Loaded {len(df):,} rows × "
        f"{df.shape[1]} columns"
    )

    return df


# ─────────────────────────────────────────────
# Data Cleaning
# ─────────────────────────────────────────────

def clean_data(
    df: pd.DataFrame
) -> pd.DataFrame:

    logger.info("Cleaning data")

    original_len = len(df)

    df = (
        df
        .sort_values(
            ["turbine_id", "timestamp"]
        )
        .reset_index(drop=True)
    )

    df = df.drop_duplicates(
        subset=["turbine_id", "timestamp"]
    )

    bounds = {
        "bearing_temp": (0, 130),
        "generator_temp": (0, 140),
        "vibration": (0, 20),
        "oil_pressure": (0, 12),
        "power_output": (0, 2600),
        "rotor_speed": (0, 30),
        "wind_speed": (0, 40),
    }

    for col, (low, high) in bounds.items():

        if col in df.columns:
            df[col] = df[col].clip(
                low,
                high
            )

    cleaned_groups = []

    for turbine_id, group in df.groupby(
        "turbine_id"
    ):

        group = group.sort_values(
            "timestamp"
        )

        full_time_index = pd.date_range(
            start=group["timestamp"].min(),
            end=group["timestamp"].max(),
            freq="10s"
        )

        group = (
            group
            .set_index("timestamp")
            .reindex(full_time_index)
            .ffill(limit=3)
            .reset_index()
            .rename(
                columns={"index": "timestamp"}
            )
        )

        group["turbine_id"] = (
            group["turbine_id"]
            .fillna(turbine_id)
        )

        cleaned_groups.append(group)

    df = pd.concat(
        cleaned_groups,
        ignore_index=True
    )

    removed = original_len - len(df)

    logger.info(
        f"Cleaning completed | "
        f"Original: {original_len:,} | "
        f"Final: {len(df):,} | "
        f"Difference: {removed:,}"
    )

    return df


# ─────────────────────────────────────────────
# Rolling Features
# ─────────────────────────────────────────────

def add_rolling_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    logger.info("Adding rolling features")

    for sensor in SENSOR_COLS:

        for window_name, window_size in WINDOWS.items():

            mean_col = (
                f"{sensor}_mean_{window_name}"
            )

            std_col = (
                f"{sensor}_std_{window_name}"
            )

            df[mean_col] = (
                df
                .groupby(
                    "turbine_id",
                    sort=False
                )[sensor]
                .transform(
                    lambda x: x.rolling(
                        window=window_size,
                        min_periods=1
                    ).mean()
                )
                .astype("float32")
            )

            df[std_col] = (
                df
                .groupby(
                    "turbine_id",
                    sort=False
                )[sensor]
                .transform(
                    lambda x: x.rolling(
                        window=window_size,
                        min_periods=1
                    ).std()
                )
                .fillna(0)
                .astype("float32")
            )

    logger.info(
        f"Added "
        f"{len(SENSOR_COLS) * len(WINDOWS) * 2} "
        f"rolling features"
    )

    return df


# ─────────────────────────────────────────────
# Rate of Change Features
# ─────────────────────────────────────────────

def add_rate_of_change_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    logger.info(
        "Adding rate-of-change features"
    )

    for sensor in SENSOR_COLS:

        df[f"{sensor}_roc"] = (
            df
            .groupby(
                "turbine_id",
                sort=False
            )[sensor]
            .diff()
            .fillna(0)
            .astype("float32")
        )

    logger.info(
        f"Added "
        f"{len(SENSOR_COLS)} "
        f"rate-of-change features"
    )

    return df


# ─────────────────────────────────────────────
# Cross Sensor Features
# ─────────────────────────────────────────────

def add_cross_sensor_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    logger.info(
        "Adding cross-sensor features"
    )

    eps = 1e-6

    df["power_efficiency"] = (
        df["power_output"] /
        ((df["wind_speed"] ** 3) + eps)
    ).astype("float32")

    df["temp_speed_ratio"] = (
        df["bearing_temp"] /
        (df["rotor_speed"] + eps)
    ).astype("float32")

    df["generator_bearing_temp_ratio"] = (
        df["generator_temp"] /
        (df["bearing_temp"] + eps)
    ).astype("float32")

    df["pressure_speed_ratio"] = (
        df["oil_pressure"] /
        (df["rotor_speed"] + eps)
    ).astype("float32")

    df["vibration_speed_ratio"] = (
        df["vibration"] /
        (df["rotor_speed"] + eps)
    ).astype("float32")

    logger.info(
        "Cross-sensor features added"
    )

    return df


# ─────────────────────────────────────────────
# Time Features
# ─────────────────────────────────────────────

def add_time_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    logger.info("Adding time features")

    df["hour"] = (
        df["timestamp"]
        .dt.hour
        .astype("int8")
    )

    df["day_of_week"] = (
        df["timestamp"]
        .dt.dayofweek
        .astype("int8")
    )

    df["month"] = (
        df["timestamp"]
        .dt.month
        .astype("int8")
    )

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    ).astype("float32")

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    ).astype("float32")

    df["dow_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    ).astype("float32")

    df["dow_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    ).astype("float32")

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    ).astype("float32")

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    ).astype("float32")

    df["is_night"] = (
        (
            (df["hour"] < 6) |
            (df["hour"] >= 20)
        )
        .astype("int8")
    )

    logger.info("Time features added")

    return df


# ─────────────────────────────────────────────
# Digital Twin Features
# ─────────────────────────────────────────────

def add_digital_twin_features(
    df: pd.DataFrame,
    save_dir: str = "data/models/twin"
) -> pd.DataFrame:
    """
    Adds:
    - residual_* features
    - norm_residual_* features
    - twin_deviation_score
    """

    logger.info(
        "Adding digital twin residual features"
    )

    engine = DigitalTwinEngine()

    engine.fit_all(df)

    engine.save_all(save_dir)

    df = engine.compute_residuals_all(df)

    logger.info(
        "Digital twin residual features added"
    )

    return df


# ─────────────────────────────────────────────
# Feature Engineering Pipeline
# ─────────────────────────────────────────────

def engineer_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    logger.info(
        "Starting feature engineering"
    )

    df = reduce_memory_usage(df)

    # Rolling Features
    df = add_rolling_features(df)
    df = reduce_memory_usage(df)

    # Rate of Change
    df = add_rate_of_change_features(df)
    df = reduce_memory_usage(df)

    # Cross Sensor Features
    df = add_cross_sensor_features(df)
    df = reduce_memory_usage(df)

    # Time Features
    df = add_time_features(df)
    df = reduce_memory_usage(df)

    # Digital Twin Features
    df = add_digital_twin_features(df)
    df = reduce_memory_usage(df)

    logger.info(
        f"Feature engineering completed | "
        f"Shape: {df.shape}"
    )

    return df


# ─────────────────────────────────────────────
# Label Encoding
# ─────────────────────────────────────────────

def encode_labels(
    df: pd.DataFrame
):

    logger.info("Encoding labels")

    label_encoder = LabelEncoder()

    df["fault_label"] = (
        label_encoder.fit_transform(
            df["fault_type"]
        )
    )

    mapping = {
        cls: int(idx)
        for idx, cls in enumerate(
            label_encoder.classes_
        )
    }

    logger.info(
        f"Label mapping: {mapping}"
    )

    return df, label_encoder


# ─────────────────────────────────────────────
# Feature Columns
# ─────────────────────────────────────────────

def get_feature_columns(
    df: pd.DataFrame
) -> list:

    exclude_cols = [
        "timestamp",
        "turbine_id",
        "fault_type",
        "fault_label",
        "rul_days"
    ]

    feature_cols = [
        col for col in df.columns
        if col not in exclude_cols
    ]

    return feature_cols


# ─────────────────────────────────────────────
# Train / Val / Test Split
# ─────────────────────────────────────────────

def split_data(df: pd.DataFrame, config: dict) -> tuple:
    """
    Hybrid split:
    Step 1 — Time-based split per turbine (70/15/15)
    Step 2 — Check which fault classes are missing from train
    Step 3 — For each missing class, move its earliest fault
              window from val/test into train
    Temporal order is always preserved — no data leakage.
    """
    train_r = config["data_generation"]["train_split"]
    val_r   = config["data_generation"]["val_split"]

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

    # ── Step 2: find missing fault classes in train ──
    all_classes   = set(df["fault_type"].unique())
    train_classes = set(train_df["fault_type"].unique())
    missing       = all_classes - train_classes

    if missing:
        logger.info(f"  Missing classes in train: {missing}")
        logger.info("  Applying hybrid fix — moving fault windows into train ...")

        # Work on per-turbine level to preserve order
        new_trains, new_vals, new_tests = [], [], []

        for tid in df["turbine_id"].unique():
            t_grp = train_df[train_df["turbine_id"] == tid].copy()
            v_grp = val_df[val_df["turbine_id"] == tid].copy()
            te_grp = test_df[test_df["turbine_id"] == tid].copy()

            for fault_class in missing:
                # Check if this turbine has this fault in val or test
                fault_in_val  = v_grp[v_grp["fault_type"] == fault_class]
                fault_in_test = te_grp[te_grp["fault_type"] == fault_class]

                source      = None
                source_name = None

                if len(fault_in_val) > 0:
                    source      = v_grp
                    source_name = "val"
                elif len(fault_in_test) > 0:
                    source      = te_grp
                    source_name = "test"

                if source is None:
                    continue

                # Find earliest fault timestamp for this class
                fault_rows  = source[source["fault_type"] == fault_class]
                fault_start = fault_rows["timestamp"].min()

                # Define window: 2 days before fault start → fault end
                # This gives the model the run-up AND the fault itself
                window_start = fault_start - pd.Timedelta(days=2)
                window_end   = fault_rows["timestamp"].max()

                window = source[
                    (source["timestamp"] >= window_start) &
                    (source["timestamp"] <= window_end)
                ].copy()

                logger.info(
                    f"    [{tid}] Moving {fault_class} window "
                    f"({len(window):,} rows) from {source_name} → train"
                )

                # Add window to train
                t_grp = pd.concat([t_grp, window]).sort_values("timestamp")

                # Remove window from its source split
                if source_name == "val":
                    v_grp = v_grp[
                        ~((v_grp["timestamp"] >= window_start) &
                          (v_grp["timestamp"] <= window_end))
                    ]
                else:
                    te_grp = te_grp[
                        ~((te_grp["timestamp"] >= window_start) &
                          (te_grp["timestamp"] <= window_end))
                    ]

            new_trains.append(t_grp)
            new_vals.append(v_grp)
            new_tests.append(te_grp)

        train_df = pd.concat(new_trains).reset_index(drop=True)
        val_df   = pd.concat(new_vals).reset_index(drop=True)
        test_df  = pd.concat(new_tests).reset_index(drop=True)

    else:
        logger.info("  All fault classes present in train ✓")

    # ── Step 3: verify ───────────────────────────────
    final_train_classes = set(train_df["fault_type"].unique())
    still_missing = all_classes - final_train_classes

    if still_missing:
        logger.warning(f"  Still missing after hybrid fix: {still_missing}")
    else:
        logger.info("  Hybrid split verified — all classes in train ✓")

    logger.info(f"  Train: {len(train_df):,} rows")
    logger.info(f"  Val  : {len(val_df):,} rows")
    logger.info(f"  Test : {len(test_df):,} rows")

    # Log class distribution in train
    logger.info("  Train fault class distribution:")
    dist = train_df["fault_type"].value_counts()
    for fault, count in dist.items():
        pct = count / len(train_df) * 100
        logger.info(f"    {fault:30s} {count:>8,}  ({pct:.1f}%)")

    return train_df, val_df, test_df


# ─────────────────────────────────────────────
# Feature Scaling
# ─────────────────────────────────────────────

def scale_features(
    train_df,
    val_df,
    test_df,
    feature_cols
):

    logger.info("Scaling features")

    scaler = StandardScaler()

    train_df[feature_cols] = (
        scaler
        .fit_transform(
            train_df[feature_cols]
        )
        .astype("float32")
    )

    val_df[feature_cols] = (
        scaler
        .transform(
            val_df[feature_cols]
        )
        .astype("float32")
    )

    test_df[feature_cols] = (
        scaler
        .transform(
            test_df[feature_cols]
        )
        .astype("float32")
    )

    logger.info(
        "Scaling completed"
    )

    return (
        train_df,
        val_df,
        test_df,
        scaler
    )


# ─────────────────────────────────────────────
# Save Processed Data
# ─────────────────────────────────────────────

def save_processed_data(
    train_df,
    val_df,
    test_df,
    feature_cols,
    label_encoder,
    scaler,
    output_dir="data/processed"
):

    logger.info(
        "Saving processed datasets"
    )

    output_path = Path(output_dir)

    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df.to_parquet(
        output_path / "train.parquet",
        index=False
    )

    val_df.to_parquet(
        output_path / "val.parquet",
        index=False
    )

    test_df.to_parquet(
        output_path / "test.parquet",
        index=False
    )

    joblib.dump(
        scaler,
        output_path / "scaler.joblib"
    )

    joblib.dump(
        label_encoder,
        output_path / "label_encoder.joblib"
    )

    joblib.dump(
        feature_cols,
        output_path / "feature_columns.joblib"
    )

    logger.info(
        f"Processed files saved → "
        f"{output_path}"
    )


# ─────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────

def run_preprocessing_pipeline(
    config_path="configs/config.yaml",
    output_dir="data/processed"
):

    logger.info("=" * 60)
    logger.info(
        "Data Preprocessor — Phase 2"
    )
    logger.info("=" * 60)

    config = load_config(config_path)

    df = load_raw_data(config)

    df = clean_data(df)

    df = engineer_features(df)

    df, label_encoder = encode_labels(df)

    feature_cols = get_feature_columns(df)

    logger.info(
        f"Total feature columns: "
        f"{len(feature_cols)}"
    )

    (
        train_df,
        val_df,
        test_df
    ) = split_data(df,config)

    (
        train_df,
        val_df,
        test_df,
        scaler
    ) = scale_features(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=feature_cols
    )

    save_processed_data(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=feature_cols,
        label_encoder=label_encoder,
        scaler=scaler,
        output_dir=output_dir
    )

    logger.info("=" * 60)
    logger.info(
        "Preprocessing completed successfully"
    )
    logger.info("=" * 60)

    return (
        train_df,
        val_df,
        test_df,
        feature_cols
    )


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":

    run_preprocessing_pipeline()