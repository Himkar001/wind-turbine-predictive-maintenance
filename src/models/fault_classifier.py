# fault_classifier.py

"""
Fault Classifier — Phase 3

XGBoost multi-class classifier.

Classes:
0: bearing_failure
1: blade_imbalance
2: electrical_fault
3: gearbox_fault
4: generator_overheating
5: maintenance
6: normal
7: oil_leak
"""

from importlib_metadata import metadata
import numpy as np
import pandas as pd
import logging
import joblib
import json

from pathlib import Path

from xgboost import XGBClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

from sklearn.utils.class_weight import compute_sample_weight


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

SAVE_DIR = Path("data/models/fault_classifier")


class FaultClassifier:
    """
    XGBoost multi-class fault classifier.

    Handles class imbalance using sample weights.

    Outputs:
    - predicted class
    - probability vector
    """

    def __init__(self, config: dict = None):

        cfg = (
            (config or {})
            .get("models", {})
            .get("fault_classification", {})
            .get("xgboost", {})
        )

        self.model = XGBClassifier(
            n_estimators=cfg.get("n_estimators", 300),
            max_depth=cfg.get("max_depth", 6),
            learning_rate=cfg.get("learning_rate", 0.05),
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            early_stopping_rounds=20,      # ← stops when val loss stops improving
            random_state=cfg.get("random_state", 42),
            n_jobs=-1,
            tree_method="hist",
        )
        self.feature_cols = None
        self.label_classes = None
        self.feature_importance = None
        self.is_fitted = False
        self.class_to_index = {}
        self.index_to_class = {}

    def _get_feature_cols(self, df: pd.DataFrame) -> list:

        exclude = {
            "timestamp",
            "turbine_id",
            "fault_type",
            "fault_label",
            "is_anomaly",
            "severity",
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
        label_classes: dict,
    ):

        self.feature_cols = self._get_feature_cols(train_df)

        self.label_classes = label_classes

        X_train = train_df[self.feature_cols].values.astype(np.float32)

        y_train = train_df["fault_label"].values.astype(int)

        X_val = val_df[self.feature_cols].values.astype(np.float32)

        y_val = val_df["fault_label"].values.astype(int)

        sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )

        logger.info("Training XGBoost classifier")
        logger.info(f"Features : {len(self.feature_cols)}")
        logger.info(f"Train    : {len(X_train):,}")
        logger.info(f"Classes  : {len(label_classes)}")

        self.model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=50,
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

        logger.info("XGBoost classifier fitted")

    def predict(self, X: np.ndarray) -> np.ndarray:

         return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        return self.model.predict_proba(X)

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:

        X = df[self.feature_cols].values.astype(np.float32)

        predictions = self.predict(X)

        probabilities = self.predict_proba(X)

        result = df.copy()

        result["predicted_fault_label"] = predictions

        result["predicted_fault_type"] = [
            self.label_classes[str(pred)]
            for pred in predictions
        ]

        result["fault_confidence"] = probabilities.max(axis=1)

        for class_id, class_name in self.label_classes.items():

            result[f"fault_proba_{class_name}"] = probabilities[:, int(class_id)]

        return result

    def evaluate(
        self,
        df: pd.DataFrame,
        split: str = "test",
    ):

        X = df[self.feature_cols].values.astype(np.float32)

        y_true = df["fault_label"].values.astype(int)

        y_pred = self.predict(X)

        accuracy = accuracy_score(
            y_true,
            y_pred,
        )

        weighted_f1 = f1_score(
            y_true,
            y_pred,
            average="weighted",
        )

        logger.info(
            f"\n=== Fault Classification — {split.upper()} ==="
        )

        logger.info(f"Accuracy : {accuracy:.4f}")

        logger.info(f"F1 Score : {weighted_f1:.4f}")

        # IMPORTANT FIX
        # only use classes actually present
        unique_labels = sorted(
            np.unique(
                np.concatenate([y_true, y_pred])
            )
        )

        class_names = [
            self.label_classes[str(label)]
            for label in unique_labels
        ]

        logger.info(
            "\n"
            + classification_report(
                y_true,
                y_pred,
                labels=unique_labels,
                target_names=class_names,
                zero_division=0,
            )
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

    def save(self, save_dir: str = str(SAVE_DIR)):

        path = Path(save_dir)

        path.mkdir(parents=True, exist_ok=True)

        self.model.save_model(
            str(path / "xgb_fault_classifier.json")
        )

        joblib.dump(
            self.feature_cols,
            path / "feature_cols.joblib",
        )

        with open(path / "metadata.json", "w") as f:

            json.dump(
                {
                    "label_classes": self.label_classes,
                    "feature_importance": dict(
                        list(self.feature_importance.items())[:20]
                    ),
                    "n_features": len(self.feature_cols),
                    "class_to_index": {str(k): v for k, v in self.class_to_index.items()},
                    "index_to_class": {str(k): v for k, v in self.index_to_class.items()},
                },
                f,
                indent=2,
            )

        logger.info(f"Fault classifier saved → {path}/")

    def load(self, save_dir: str = str(SAVE_DIR)):

        path = Path(save_dir)

        self.model.load_model(
            str(path / "xgb_fault_classifier.json")
        )

        self.feature_cols = joblib.load(
            path / "feature_cols.joblib"
        )

        with open(path / "metadata.json") as f:

            metadata = json.load(f)

        self.label_classes = metadata["label_classes"]

        self.feature_importance = metadata["feature_importance"]
        self.class_to_index = {int(k): v for k, v in metadata["class_to_index"].items()}
        self.index_to_class = {int(k): v for k, v in metadata["index_to_class"].items()}

        self.is_fitted = True

        logger.info(f"Fault classifier loaded ← {path}/")


def run_fault_classification_pipeline(
    train_path: str = "data/processed/train.parquet",
    val_path: str = "data/processed/val.parquet",
    test_path: str = "data/processed/test.parquet"
):

    logger.info("=" * 60)
    logger.info("Fault Classification Pipeline — Phase 3")
    logger.info("=" * 60)

    train_df = pd.read_parquet(train_path)

    val_df = pd.read_parquet(val_path)

    test_df = pd.read_parquet(test_path)

    import joblib

    label_encoder = joblib.load(
        "data/processed/label_encoder.joblib"
    )

    label_classes = {
        str(i): cls
        for i, cls in enumerate(label_encoder.classes_)
    }

    logger.info(
        f"Train: {len(train_df):,} | "
        f"Val: {len(val_df):,} | "
        f"Test: {len(test_df):,}"
    )

    logger.info(f"Classes: {label_classes}")

    classifier = FaultClassifier()

    classifier.fit(
        train_df=train_df,
        val_df=val_df,
        label_classes=label_classes,
    )

    classifier.evaluate(
        val_df,
        split="validation",
    )

    classifier.evaluate(
        test_df,
        split="test",
    )

    classifier.save()

    logger.info("=" * 60)
    logger.info("Fault Classification complete")
    logger.info("=" * 60)

    return classifier


if __name__ == "__main__":

    run_fault_classification_pipeline()