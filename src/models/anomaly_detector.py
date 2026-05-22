# anomaly_detector.py

"""
Anomaly Detector — Phase 3

Two-stage anomaly detection:
1. Isolation Forest
2. LSTM Autoencoder

Final anomaly score = weighted combination of both.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import logging
import joblib
import json
from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
)

from torch.utils.data import DataLoader, TensorDataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

SAVE_DIR = Path("data/models/anomaly")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.enc_fc = nn.Linear(hidden_dim, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, hidden_dim)

        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        _, (hidden, _) = self.encoder(x)

        latent = self.enc_fc(hidden[-1])

        dec_input = self.dec_fc(latent)
        dec_input = dec_input.unsqueeze(1).repeat(1, seq_len, 1)

        dec_out, _ = self.decoder(dec_input)

        reconstructed = self.output_fc(dec_out)

        return reconstructed

    def reconstruction_error(self, x):
        with torch.no_grad():
            reconstructed = self.forward(x)
            error = torch.mean((x - reconstructed) ** 2, dim=(1, 2))

        return error.cpu().numpy()


class IsolationForestDetector:
    def __init__(
        self,
        contamination: float = 0.08,
        n_estimators: int = 200,
        random_state: int = 42,
    ):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )

        self.is_fitted = False

    def fit(self, X: np.ndarray):
        logger.info(f"Fitting Isolation Forest on {X.shape[0]:,} samples")

        self.model.fit(X)
        self.is_fitted = True

        logger.info("Isolation Forest fitted")

    def score(self, X: np.ndarray) -> np.ndarray:
        raw_scores = self.model.decision_function(X)

        flipped_scores = -raw_scores

        normalized_scores = (
            flipped_scores - flipped_scores.min()
        ) / (flipped_scores.max() - flipped_scores.min() + 1e-8)

        return normalized_scores.astype(np.float32)

    def save(self, path: Path):
        joblib.dump(self.model, path / "isolation_forest.joblib")

    def load(self, path: Path):
        self.model = joblib.load(path / "isolation_forest.joblib")
        self.is_fitted = True


class LSTMAutoencoderDetector:
    def __init__(
        self,
        input_dim: int,
        seq_len: int = 60,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        epochs: int = 30,
        batch_size: int = 256,
        lr: float = 1e-3,
    ):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.threshold = None

        self.model = LSTMAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        ).to(DEVICE)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def _make_sequences(self, X: np.ndarray) -> torch.Tensor:
        n = len(X) - self.seq_len + 1

        sequences = np.stack(
            [X[i : i + self.seq_len] for i in range(0, n, self.seq_len)]
        )

        return torch.FloatTensor(sequences)

    def fit(self, X_normal: np.ndarray):
        logger.info(
            f"Training LSTM Autoencoder on {X_normal.shape[0]:,} normal samples"
        )
        logger.info(f"Device: {DEVICE}")

        sequences = self._make_sequences(X_normal)

        dataset = TensorDataset(sequences)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

        self.model.train()

        for epoch in range(self.epochs):
            total_loss = 0.0

            for (batch,) in loader:
                batch = batch.to(DEVICE)

                self.optimizer.zero_grad()

                reconstructed = self.model(batch)

                loss = self.criterion(reconstructed, batch)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(loader)

            if (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{self.epochs} | Loss: {avg_loss:.6f}"
                )

        self.model.eval()

        errors = self._get_errors(X_normal)

        self.threshold = float(np.percentile(errors, 95))

        logger.info(f"LSTM AE threshold set at: {self.threshold:.6f}")
        logger.info("LSTM Autoencoder trained")

    def _get_errors(self, X: np.ndarray) -> np.ndarray:
        sequences = self._make_sequences(X)

        loader = DataLoader(
            TensorDataset(sequences),
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.model.eval()

        all_errors = []

        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(DEVICE)

                errors = self.model.reconstruction_error(batch)

                all_errors.extend(errors.tolist())

        repeated_errors = np.repeat(all_errors, self.seq_len)

        if len(repeated_errors) > len(X):
            repeated_errors = repeated_errors[: len(X)]

        elif len(repeated_errors) < len(X):
            repeated_errors = np.pad(
                repeated_errors,
                (0, len(X) - len(repeated_errors)),
                constant_values=repeated_errors[-1],
            )

        return repeated_errors.astype(np.float32)

    def score(self, X: np.ndarray) -> np.ndarray:
        errors = self._get_errors(X)

        scores = np.clip(errors / (2 * self.threshold + 1e-8), 0, 1)

        return scores.astype(np.float32)

    def save(self, path: Path):
        torch.save(self.model.state_dict(), path / "lstm_ae.pt")

        with open(path / "lstm_ae_meta.json", "w") as f:
            json.dump(
                {
                    "threshold": self.threshold,
                    "seq_len": self.seq_len,
                    "input_dim": self.model.input_dim,
                    "hidden_dim": self.model.hidden_dim,
                    "latent_dim": self.model.latent_dim,
                },
                f,
                indent=2,
            )

    def load(self, path: Path):
        with open(path / "lstm_ae_meta.json") as f:
            meta = json.load(f)

        self.threshold = meta["threshold"]
        self.seq_len = meta["seq_len"]

        self.model = LSTMAutoencoder(
            input_dim=meta["input_dim"],
            hidden_dim=meta["hidden_dim"],
            latent_dim=meta["latent_dim"],
        ).to(DEVICE)

        self.model.load_state_dict(
            torch.load(path / "lstm_ae.pt", map_location=DEVICE)
        )

        self.model.eval()


class AnomalyDetector:
    def __init__(
        self,
        input_dim: int,
        seq_len: int = 60,
        if_weight: float = 0.4,
        lstm_weight: float = 0.6,
    ):
        self.if_detector = IsolationForestDetector()

        self.lstm_detector = LSTMAutoencoderDetector(
            input_dim=input_dim,
            seq_len=seq_len,
        )

        self.if_weight = if_weight
        self.lstm_weight = lstm_weight

        self.best_threshold = 0.5
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame, feature_cols: list):
        X = train_df[feature_cols].values.astype(np.float32)

        X_normal = train_df[train_df["fault_type"] == "normal"][
            feature_cols
        ].values.astype(np.float32)

        logger.info(
            f"Training anomaly detector on {X.shape[0]:,} total samples "
            f"and {X_normal.shape[0]:,} normal samples"
        )

        self.if_detector.fit(X)

        self.lstm_detector.fit(X_normal)

        self.is_fitted = True

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        if_scores = self.if_detector.score(X)

        lstm_scores = self.lstm_detector.score(X)

        combined_scores = (
            self.if_weight * if_scores + self.lstm_weight * lstm_scores
        )

        return combined_scores.astype(np.float32)

    def tune_threshold(self, val_df: pd.DataFrame, feature_cols: list):
        """
        Find optimal threshold on validation set using F1 score.
        """

        logger.info("Tuning anomaly threshold on validation set")

        X = val_df[feature_cols].values.astype(np.float32)

        # FORCE BINARY LABELS
        y_true = val_df["is_anomaly"].astype(int).values

        scores = self.predict_scores(X)

        precisions, recalls, thresholds = precision_recall_curve(
            y_true,
            scores
        )

        f1_scores = (
            2 * precisions * recalls /
            (precisions + recalls + 1e-8)
        )

        best_idx = np.argmax(f1_scores)

        # thresholds array is 1 shorter than precision/recall
        if best_idx >= len(thresholds):
            best_idx = len(thresholds) - 1

        self.best_threshold = float(thresholds[best_idx])

        logger.info(
            f"Best threshold: {self.best_threshold:.4f} | "
            f"F1: {f1_scores[best_idx]:.4f}"
    )

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.predict_scores(X)

        return (scores >= self.best_threshold).astype(int)

    def evaluate(self, df: pd.DataFrame, feature_cols: list, split: str = "test"):
        X = df[feature_cols].values.astype(np.float32)

        y_true = df["is_anomaly"].astype(int).values

        scores = self.predict_scores(X)

        y_pred = (scores >= self.best_threshold).astype(int)

        logger.info(f"\n=== Anomaly Detection Evaluation: {split.upper()} ===")

        logger.info(f"ROC-AUC: {roc_auc_score(y_true, scores):.4f}")

        logger.info(f"F1 Score: {f1_score(y_true, y_pred):.4f}")

        logger.info(
            "\n"
            + classification_report(
                y_true,
                y_pred,
                target_names=["Normal", "Anomaly"],
            )
        )

        return scores, y_pred

    def save(self, save_dir: str = str(SAVE_DIR)):
        path = Path(save_dir)

        path.mkdir(parents=True, exist_ok=True)

        self.if_detector.save(path)

        self.lstm_detector.save(path)

        with open(path / "detector_meta.json", "w") as f:
            json.dump(
                {
                    "if_weight": self.if_weight,
                    "lstm_weight": self.lstm_weight,
                    "best_threshold": self.best_threshold,
                },
                f,
                indent=2,
            )

        logger.info(f"Anomaly detector saved to {path}")

    def load(self, save_dir: str = str(SAVE_DIR)):
        path = Path(save_dir)

        with open(path / "detector_meta.json") as f:
            meta = json.load(f)

        self.if_weight = meta["if_weight"]
        self.lstm_weight = meta["lstm_weight"]
        self.best_threshold = meta["best_threshold"]

        self.if_detector.load(path)

        self.lstm_detector.load(path)

        self.is_fitted = True

        logger.info(f"Anomaly detector loaded from {path}")


def run_anomaly_detection_pipeline(
    train_path: str = "data/processed/train.parquet",
    val_path: str = "data/processed/val.parquet",
    test_path: str = "data/processed/test.parquet",
):
    logger.info("=" * 60)
    logger.info("Anomaly Detection Pipeline — Phase 3")
    logger.info("=" * 60)

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    exclude_cols = {
        "timestamp",
        "turbine_id",
        "fault_type",
        "fault_label",
        "is_anomaly",
        "severity",
    }

    feature_cols = [
        col
        for col in train_df.columns
        if col not in exclude_cols
        and train_df[col].dtype
        in [
            np.float32,
            np.float64,
            np.int32,
            np.int64,
        ]
    ]

    logger.info(f"Number of features: {len(feature_cols)}")
    logger.info(
        f"Train: {len(train_df):,} | "
        f"Val: {len(val_df):,} | "
        f"Test: {len(test_df):,}"
    )

    detector = AnomalyDetector(input_dim=len(feature_cols))

    detector.fit(train_df, feature_cols)

    detector.tune_threshold(val_df, feature_cols)

    detector.evaluate(val_df, feature_cols, split="validation")
    detector.evaluate(test_df, feature_cols, split="test")

    detector.save()

    logger.info("=" * 60)
    logger.info("Anomaly Detection complete")
    logger.info("=" * 60)

    return detector


if __name__ == "__main__":
    run_anomaly_detection_pipeline()