# rul_predictor.py

"""
RUL Predictor — Phase 3

Remaining Useful Life prediction using XGBoost Regressor.

Output:
- predicted RUL in days
- urgency category
"""

import numpy as np
import pandas as pd
import logging
import joblib
import json

from pathlib import Path

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

SAVE_DIR = Path("data/models/rul_predictor")


URGENCY_THRESHOLDS = {
    "critical": 3,
    "high": 7,
    "medium": 14,
    "low": float("inf"),
}


def compute_rul_labels(
    df: pd.DataFrame,
    interval_seconds: int = 10,
) -> pd.DataFrame:

    logger.info("Computing RUL labels")

    results = []

    for turbine_id, group in df.groupby("turbine_id"):

        group = group.sort_values("timestamp").copy()

        group["rul_days"] = 30.0

        is_fault = (
            (group["fault_type"] != "normal")
            &
            (group["fault_type"] != "maintenance")
        )

        fault_starts = group[
            is_fault &
            (~is_fault.shift(1, fill_value=False))
        ]["timestamp"].values

        if len(fault_starts) == 0:

            results.append(group)

            continue

        timestamps = group["timestamp"].values

        for idx, timestamp in enumerate(timestamps):

            future_faults = fault_starts[fault_starts > timestamp]

            if len(future_faults) > 0:

                next_fault = future_faults[0]

                delta_ns = (
                    next_fault - timestamp
                ).astype("int64")

                delta_days = delta_ns / (1e9 * 86400)

                rul = min(float(delta_days), 30.0)

                group.iloc[
                    idx,
                    group.columns.get_loc("rul_days")
                ] = max(rul, 0.0)

        results.append(group)

    final_df = pd.concat(results).reset_index(drop=True)

    logger.info(
        f"RUL range: "
        f"{final_df['rul_days'].min():.2f} → "
        f"{final_df['rul_days'].max():.2f}"
    )

    logger.info(
        f"Mean RUL: {final_df['rul_days'].mean():.2f}"
    )

    return final_df


def get_urgency(rul_days: float) -> str:

    if rul_days < URGENCY_THRESHOLDS["critical"]:
        return "critical"

    elif rul_days < URGENCY_THRESHOLDS["high"]:
        return "high"

    elif rul_days < URGENCY_THRESHOLDS["medium"]:
        return "medium"

    else:
        return "low"


class RULPredictor:
    """
    XGBoost Regressor for RUL prediction.
    """

    def __init__(self, config: dict = None):

        cfg = (
            (config or {})
            .get("models", {})
            .get("rul_prediction", {})
            .get("xgboost", {})
        )

        self.model = XGBRegressor(
            n_estimators=cfg.get("n_estimators", 300),
            max_depth=cfg.get("max_depth", 6),
            learning_rate=cfg.get("learning_rate", 0.05),
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=cfg.get("random_state", 42),
            n_jobs=-1,
            tree_method="hist",
        )

        self.feature_cols = None

        self.feature_importance = None

        self.train_mae = None

        self.is_fitted = False

    def _get_feature_cols(
        self,
        df: pd.DataFrame,
    ) -> list:

        exclude = {
            "timestamp",
            "turbine_id",
            "fault_type",
            "fault_label",
            "is_anomaly",
            "severity",
            "rul_days",
        }

        feature_cols = [
            col
            for col in df.columns
            if col not in exclude
            and df[col].dtype
            in [
                np.float32,
                np.float64,
                np.int32,
                np.int64,
            ]
        ]

        return feature_cols

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ):

        logger.info("Computing RUL labels for train set")

        train_df = compute_rul_labels(train_df)

        val_df = compute_rul_labels(val_df)

        self.feature_cols = self._get_feature_cols(train_df)

        X_train = train_df[self.feature_cols].values.astype(np.float32)

        y_train = train_df["rul_days"].values.astype(np.float32)

        X_val = val_df[self.feature_cols].values.astype(np.float32)

        y_val = val_df["rul_days"].values.astype(np.float32)

        logger.info("Training XGBoost RUL Regressor")

        logger.info(f"Features : {len(self.feature_cols)}")

        logger.info(f"Train    : {len(X_train):,}")

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )

        y_pred_train = self.model.predict(X_train)

        self.train_mae = mean_absolute_error(
            y_train,
            y_pred_train,
        )

        importance = self.model.feature_importances_

        self.feature_importance = dict(
            sorted(
                zip(self.feature_cols, importance.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        self.is_fitted = True

        logger.info(
            f"Train MAE: {self.train_mae:.4f} days"
        )

        logger.info("RUL Predictor fitted")

        return train_df, val_df

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:

        predictions = self.model.predict(X)

        predictions = np.clip(
            predictions,
            0,
            30,
        )

        return predictions.astype(np.float32)

    def predict_df(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        X = df[self.feature_cols].values.astype(np.float32)

        predictions = self.predict(X)

        result = df.copy()

        result["predicted_rul_days"] = predictions

        result["rul_urgency"] = [
            get_urgency(rul)
            for rul in predictions
        ]

        return result

    def evaluate(
        self,
        df: pd.DataFrame,
        split: str = "test",
    ):

        df = compute_rul_labels(df)

        X = df[self.feature_cols].values.astype(np.float32)

        y_true = df["rul_days"].values.astype(np.float32)

        y_pred = self.predict(X)

        mae = mean_absolute_error(
            y_true,
            y_pred,
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )

        r2 = r2_score(
            y_true,
            y_pred,
        )

        logger.info(
            f"\n=== RUL Prediction — {split.upper()} ==="
        )

        logger.info(f"MAE  : {mae:.4f} days")

        logger.info(f"RMSE : {rmse:.4f} days")

        logger.info(f"R²   : {r2:.4f}")

        urgencies = [
            get_urgency(rul)
            for rul in y_pred
        ]

        for urgency in [
            "critical",
            "high",
            "medium",
            "low",
        ]:

            count = urgencies.count(urgency)

            percentage = (
                count / len(urgencies)
            ) * 100

            logger.info(
                f"{urgency:10s}: "
                f"{count:>8,} "
                f"({percentage:.1f}%)"
            )

        logger.info("Top 10 important features:")

        top_features = list(
            self.feature_importance.items()
        )[:10]

        for idx, (feature, importance) in enumerate(top_features):

            logger.info(
                f"{idx + 1:2d}. "
                f"{feature:35s} "
                f"{importance:.4f}"
            )

        return y_pred

    def save(
        self,
        save_dir: str = str(SAVE_DIR),
    ):

        path = Path(save_dir)

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model.save_model(
            str(path / "xgb_rul_predictor.json")
        )

        joblib.dump(
            self.feature_cols,
            path / "feature_cols.joblib",
        )

        with open(path / "metadata.json", "w") as f:

            json.dump(
                {
                    "train_mae": self.train_mae,
                    "urgency_thresholds": URGENCY_THRESHOLDS,
                    "feature_importance": dict(
                        list(self.feature_importance.items())[:20]
                    ),
                    "n_features": len(self.feature_cols),
                },
                f,
                indent=2,
                default=str,
            )

        logger.info(
            f"RUL predictor saved → {path}/"
        )

    def load(
        self,
        save_dir: str = str(SAVE_DIR),
    ):

        path = Path(save_dir)

        self.model.load_model(
            str(path / "xgb_rul_predictor.json")
        )

        self.feature_cols = joblib.load(
            path / "feature_cols.joblib"
        )

        with open(path / "metadata.json") as f:

            metadata = json.load(f)

        self.train_mae = metadata["train_mae"]

        self.feature_importance = metadata["feature_importance"]

        self.is_fitted = True

        logger.info(
            f"RUL predictor loaded ← {path}/"
        )


def run_rul_pipeline(
    train_path: str = "data/processed/train.parquet",
    val_path: str = "data/processed/val.parquet",
    test_path: str = "data/processed/test.parquet",
):

    logger.info("=" * 60)

    logger.info("RUL Prediction Pipeline — Phase 3")

    logger.info("=" * 60)

    train_df = pd.read_parquet(train_path)

    val_df = pd.read_parquet(val_path)

    test_df = pd.read_parquet(test_path)

    logger.info(
        f"Train: {len(train_df):,} | "
        f"Val: {len(val_df):,} | "
        f"Test: {len(test_df):,}"
    )

    predictor = RULPredictor()

    predictor.fit(
        train_df,
        val_df,
    )

    predictor.evaluate(
        val_df,
        split="validation",
    )

    predictor.evaluate(
        test_df,
        split="test",
    )

    predictor.save()

    logger.info("=" * 60)

    logger.info("RUL Prediction complete")

    logger.info("=" * 60)

    return predictor


if __name__ == "__main__":

    run_rul_pipeline()