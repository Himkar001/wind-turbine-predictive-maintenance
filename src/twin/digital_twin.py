"""
Digital Twin Engine — Phase 2
==============================

This file builds a Digital Twin for each wind turbine.

The Digital Twin learns the expected healthy behavior of each turbine using
normal sensor readings. Then, for every new sensor reading, it predicts what
the healthy value should have been and compares it with the actual value.

Residual = actual_value - expected_value

Large residual means the turbine is behaving differently from its healthy
baseline, which can be used as an anomaly signal.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# Sensors that the digital twin will model
TARGET_SENSORS = [
    "bearing_temp",
    "generator_temp",
    "vibration",
    "oil_pressure",
    "power_output"
]


# Features used to predict expected healthy sensor behavior
TWIN_INPUT_FEATURES = [
    "wind_speed",
    "rotor_speed",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos"
]


class TurbineDigitalTwin:
    """
    Digital twin for a single turbine.

    It trains one regression model per target sensor.
    Example:
        expected_bearing_temp = f(wind_speed, rotor_speed, hour_sin, ...)
    """

    def __init__(self, turbine_id: str):
        self.turbine_id = turbine_id
        self.models = {}
        self.baselines = {}
        self.residual_stds = {}
        self.metrics = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        """
        Train the digital twin using healthy readings only.

        Parameters
        ----------
        df : pd.DataFrame
            Training data for one turbine only.
        """

        healthy = df[df["fault_type"] == "normal"].copy()

        logger.info(
            f"[{self.turbine_id}] Training twin on "
            f"{len(healthy):,} healthy readings"
        )

        if len(healthy) < 100:
            logger.warning(
                f"[{self.turbine_id}] Very few healthy readings. "
                "Twin accuracy may be low."
            )

        X = healthy[TWIN_INPUT_FEATURES].values

        for sensor in TARGET_SENSORS:
            y = healthy[sensor].values

            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )

            model.fit(X, y)

            y_pred = model.predict(X)
            residuals = y - y_pred

            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            self.models[sensor] = model
            self.baselines[sensor] = float(np.mean(y))
            self.residual_stds[sensor] = float(np.std(residuals))
            self.metrics[sensor] = {
                "mae": round(mae, 4),
                "r2": round(r2, 4)
            }

            logger.info(
                f"[{self.turbine_id}] {sensor:20s} "
                f"MAE={mae:.4f} | R2={r2:.4f}"
            )

        self.is_fitted = True
        logger.info(f"[{self.turbine_id}] Digital twin trained successfully")

    def predict_expected(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict expected healthy sensor values.

        Returns a dataframe with expected_<sensor> columns.
        """

        if not self.is_fitted:
            raise RuntimeError(
                f"Digital twin for {self.turbine_id} is not fitted yet."
            )

        X = df[TWIN_INPUT_FEATURES].values
        result = df.copy()

        for sensor in TARGET_SENSORS:
            result[f"expected_{sensor}"] = self.models[sensor].predict(X)

        return result

    def compute_residuals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute actual - expected residuals for each sensor.

        Also computes normalized residuals and one combined
        twin_deviation_score.
        """

        df = self.predict_expected(df)

        norm_residual_cols = []

        for sensor in TARGET_SENSORS:
            residual_col = f"residual_{sensor}"
            expected_col = f"expected_{sensor}"
            norm_col = f"norm_residual_{sensor}"

            df[residual_col] = df[sensor] - df[expected_col]

            std = self.residual_stds[sensor]
            df[norm_col] = df[residual_col] / (std + 1e-8)

            norm_residual_cols.append(norm_col)

        df["twin_deviation_score"] = np.sqrt(
            df[norm_residual_cols].pow(2).mean(axis=1)
        )

        return df

    def save(self, save_dir: str = "data/models/twin"):
        """
        Save all models and metadata for this turbine.
        """

        path = Path(save_dir) / self.turbine_id
        path.mkdir(parents=True, exist_ok=True)

        for sensor, model in self.models.items():
            joblib.dump(model, path / f"{sensor}_model.joblib")

        metadata = {
            "turbine_id": self.turbine_id,
            "baselines": self.baselines,
            "residual_stds": self.residual_stds,
            "metrics": self.metrics,
            "target_sensors": TARGET_SENSORS,
            "input_features": TWIN_INPUT_FEATURES
        }

        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"[{self.turbine_id}] Twin saved to {path}")

    @classmethod
    def load(
        cls,
        turbine_id: str,
        save_dir: str = "data/models/twin"
    ) -> "TurbineDigitalTwin":
        """
        Load a saved digital twin.
        """

        path = Path(save_dir) / turbine_id

        twin = cls(turbine_id)

        with open(path / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        twin.baselines = metadata["baselines"]
        twin.residual_stds = metadata["residual_stds"]
        twin.metrics = metadata["metrics"]

        for sensor in metadata["target_sensors"]:
            twin.models[sensor] = joblib.load(path / f"{sensor}_model.joblib")

        twin.is_fitted = True

        logger.info(f"[{turbine_id}] Twin loaded from {path}")

        return twin


class DigitalTwinEngine:
    """
    Manages digital twins for all turbines.
    """

    def __init__(self):
        self.twins = {}

    def fit_all(self, train_df: pd.DataFrame):
        """
        Train one digital twin for each turbine.
        """

        logger.info("Training digital twins for all turbines")

        turbine_ids = train_df["turbine_id"].unique()

        for turbine_id in turbine_ids:
            turbine_data = train_df[train_df["turbine_id"] == turbine_id]

            twin = TurbineDigitalTwin(turbine_id)
            twin.fit(turbine_data)

            self.twins[turbine_id] = twin

        logger.info(f"Trained {len(self.twins)} turbine digital twins")

    def compute_residuals_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute residuals for all turbines.
        """

        results = []

        for turbine_id, group in df.groupby("turbine_id"):
            if turbine_id not in self.twins:
                logger.warning(f"No twin found for {turbine_id}. Skipping.")
                continue

            result = self.twins[turbine_id].compute_residuals(group)
            results.append(result)

        if not results:
            raise ValueError("No residuals were computed. Check turbine IDs.")

        final_df = pd.concat(results, ignore_index=True)

        if "timestamp" in final_df.columns:
            final_df = final_df.sort_values(
                ["turbine_id", "timestamp"]
            ).reset_index(drop=True)

        return final_df

    def save_all(self, save_dir: str = "data/models/twin"):
        """
        Save all turbine twins.
        """

        for twin in self.twins.values():
            twin.save(save_dir)

        logger.info(f"All digital twins saved to {save_dir}")

    def load_all(
        self,
        turbine_ids: list,
        save_dir: str = "data/models/twin"
    ):
        """
        Load digital twins for multiple turbines.
        """

        for turbine_id in turbine_ids:
            self.twins[turbine_id] = TurbineDigitalTwin.load(
                turbine_id=turbine_id,
                save_dir=save_dir
            )

        logger.info(f"Loaded {len(self.twins)} digital twins")

    def get_deviation_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create per-turbine deviation summary.

        Useful for dashboard:
        - average deviation
        - maximum deviation
        - deviation volatility
        """

        residual_df = self.compute_residuals_all(df)

        summary = (
            residual_df
            .groupby("turbine_id")["twin_deviation_score"]
            .agg(["mean", "max", "std"])
            .round(4)
            .rename(
                columns={
                    "mean": "avg_deviation",
                    "max": "max_deviation",
                    "std": "deviation_volatility"
                }
            )
            .reset_index()
        )

        return summary


def run_digital_twin_pipeline(
    train_path: str = "data/processed/train.parquet",
    save_dir: str = "data/models/twin"
):
    """
    Main pipeline to train and save digital twins.
    """

    logger.info("=" * 60)
    logger.info("Digital Twin Engine — Phase 2")
    logger.info("=" * 60)

    train_df = pd.read_parquet(train_path)

    logger.info(f"Loaded training data: {len(train_df):,} rows")

    engine = DigitalTwinEngine()
    engine.fit_all(train_df)
    engine.save_all(save_dir)

    logger.info("Deviation score summary on training data:")

    summary = engine.get_deviation_summary(train_df)

    for _, row in summary.iterrows():
        logger.info(
            f"{row['turbine_id']} | "
            f"avg={row['avg_deviation']:.4f} | "
            f"max={row['max_deviation']:.4f} | "
            f"volatility={row['deviation_volatility']:.4f}"
        )

    logger.info("=" * 60)
    logger.info("Digital Twin Engine completed successfully")
    logger.info("=" * 60)

    return engine


if __name__ == "__main__":
    run_digital_twin_pipeline()