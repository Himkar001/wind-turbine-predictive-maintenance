# model_pipeline.py

"""
Unified Model Pipeline — Phase 3
================================

Runs all three models together:

1. Anomaly Detection
2. Fault Classification
3. RUL Prediction

Final Output:
-------------
{
    anomaly_score,
    is_anomaly,

    predicted_fault_label,
    predicted_fault_type,
    fault_confidence,

    predicted_rul_days,
    rul_urgency
}

This file acts as the main inference engine
for the predictive maintenance platform.
"""

import numpy as np
import pandas as pd
import logging
import json

from pathlib import Path

from src.models.anomaly_detector import AnomalyDetector
from src.models.fault_classifier import FaultClassifier
from src.models.rul_predictor import RULPredictor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


class PredictiveMaintenancePipeline:
    """
    Unified pipeline combining:

    - anomaly detector
    - fault classifier
    - RUL predictor
    """

    def __init__(self):

        self.anomaly_detector = None

        self.fault_classifier = None

        self.rul_predictor = None

        self.is_loaded = False

    def load_models(self):

        logger.info("=" * 60)
        logger.info("Loading Phase 3 Models")
        logger.info("=" * 60)

        # ─────────────────────────────────────
        # Load anomaly detector
        # ─────────────────────────────────────

        self.anomaly_detector = AnomalyDetector(
            input_dim=87
        )

        self.anomaly_detector.load()

        # ─────────────────────────────────────
        # Load fault classifier
        # ─────────────────────────────────────

        self.fault_classifier = FaultClassifier()

        self.fault_classifier.load()

        # ─────────────────────────────────────
        # Load RUL predictor
        # ─────────────────────────────────────

        self.rul_predictor = RULPredictor()

        self.rul_predictor.load()

        self.is_loaded = True

        logger.info("=" * 60)
        logger.info("All models loaded successfully")
        logger.info("=" * 60)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs all 3 models together.

        Input:
        ------
        DataFrame with processed features

        Output:
        -------
        DataFrame with:
        - anomaly prediction
        - fault prediction
        - RUL prediction
        """

        if not self.is_loaded:

            raise RuntimeError(
                "Models not loaded. Call load_models() first."
            )

        logger.info("=" * 60)
        logger.info("Running Unified Prediction Pipeline")
        logger.info("=" * 60)

        result_df = df.copy()

        # =========================================================
        # STEP 1 — ANOMALY DETECTION
        # =========================================================

        logger.info("Step 1 — Anomaly Detection")

        feature_cols = self.fault_classifier.feature_cols

        X = result_df[feature_cols].values.astype(np.float32)

        anomaly_scores = (
            self.anomaly_detector.predict_scores(X)
        )

        anomaly_preds = (
            anomaly_scores >=
            self.anomaly_detector.best_threshold
        ).astype(int)

        result_df["anomaly_score"] = anomaly_scores

        result_df["is_anomaly_prediction"] = anomaly_preds

        logger.info("Anomaly detection complete")

        # =========================================================
        # STEP 2 — FAULT CLASSIFICATION
        # =========================================================

        logger.info("Step 2 — Fault Classification")

        fault_result = self.fault_classifier.predict_df(
            result_df
        )

        result_df["predicted_fault_label"] = (
            fault_result["predicted_fault_label"]
        )

        result_df["predicted_fault_type"] = (
            fault_result["predicted_fault_type"]
        )

        result_df["fault_confidence"] = (
            fault_result["fault_confidence"]
        )

        # Copy all probability columns
        proba_cols = [
            col
            for col in fault_result.columns
            if col.startswith("fault_proba_")
        ]

        for col in proba_cols:

            result_df[col] = fault_result[col]

        logger.info("Fault classification complete")

        # =========================================================
        # STEP 3 — RUL PREDICTION
        # =========================================================

        logger.info("Step 3 — RUL Prediction")

        rul_result = self.rul_predictor.predict_df(
            result_df
        )

        result_df["predicted_rul_days"] = (
            rul_result["predicted_rul_days"]
        )

        result_df["rul_urgency"] = (
            rul_result["rul_urgency"]
        )

        logger.info("RUL prediction complete")

        # =========================================================
        # STEP 4 — OVERALL RISK SCORE
        # =========================================================

        logger.info("Step 4 — Computing Overall Risk Score")

        result_df["overall_risk_score"] = (
            0.5 * result_df["anomaly_score"]
            +
            0.3 * result_df["fault_confidence"]
            +
            0.2 * (
                1 -
                (result_df["predicted_rul_days"] / 30.0)
            )
        )

        result_df["overall_risk_score"] = (
            result_df["overall_risk_score"]
            .clip(0, 1)
        )

        logger.info("Risk scoring complete")

        logger.info("=" * 60)
        logger.info("Unified pipeline complete")
        logger.info("=" * 60)

        return result_df

    def predict_single(
        self,
        row: pd.DataFrame,
    ) -> dict:
        """
        Predict for a single sensor reading.

        Returns dictionary output.
        """

        prediction_df = self.predict(row)

        result = prediction_df.iloc[0]

        output = {
            "anomaly_score":
                float(result["anomaly_score"]),

            "is_anomaly_prediction":
                int(result["is_anomaly_prediction"]),

            "predicted_fault_label":
                int(result["predicted_fault_label"]),

            "predicted_fault_type":
                str(result["predicted_fault_type"]),

            "fault_confidence":
                float(result["fault_confidence"]),

            "predicted_rul_days":
                float(result["predicted_rul_days"]),

            "rul_urgency":
                str(result["rul_urgency"]),

            "overall_risk_score":
                float(result["overall_risk_score"]),
        }

        return output


# ───────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────

def run_pipeline_demo(
    sample_path: str = "data/processed/test.parquet",
):

    logger.info("=" * 60)
    logger.info("Unified Pipeline Demo")
    logger.info("=" * 60)

    df = pd.read_parquet(sample_path)

    sample_df = df.head(1000).copy()

    pipeline = PredictiveMaintenancePipeline()

    pipeline.load_models()

    results = pipeline.predict(sample_df)

    logger.info(results.head())

    # Save demo output
    output_path = Path("outputs")

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_file = output_path / "pipeline_predictions.parquet"

    results.to_parquet(save_file)

    logger.info(
        f"Predictions saved → {save_file}"
    )

    logger.info("=" * 60)
    logger.info("Pipeline demo complete")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":

    run_pipeline_demo()