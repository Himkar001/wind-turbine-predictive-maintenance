"""
PyTorch Dataset — Phase 2
==========================

Prepares sequential time-series windows for:
- LSTM
- GRU
- Transformers
- Temporal anomaly detection
- RUL prediction

This converts tabular turbine sensor data into sequences.

Example:
    Input:
        t1, t2, t3, t4, t5

    Output:
        sequence_length = 5
        X = [t1,t2,t3,t4,t5]
        y = label_at_t5

Supports:
- Classification
- Regression (RUL prediction)
- Sequence anomaly detection
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
# Dataset Class
# ─────────────────────────────────────────────────────

class TurbineSequenceDataset(Dataset):
    """
    Creates sequential windows for turbine sensor data.

    Example:
        sequence_length = 60

        X:
            [60 timesteps × num_features]

        y:
            label of final timestep
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        feature_cols: list,
        target_col: str = "fault_label",
        sequence_length: int = 60,
        stride: int = 1,
        task: str = "classification"
    ):

        self.df = dataframe
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.sequence_length = sequence_length
        self.stride = stride
        self.task = task

        logger.info("Preparing sequence dataset")

        self.X = []
        self.y = []

        self._build_sequences()

        self.X = np.array(self.X, dtype=np.float32)

        if self.task == "classification":
            self.y = np.array(self.y, dtype=np.int64)

        else:
            self.y = np.array(self.y, dtype=np.float32)

        logger.info(
            f"Dataset created | "
            f"Sequences: {len(self.X):,} | "
            f"Shape: {self.X.shape}"
        )

    def _build_sequences(self):

        total_sequences = 0

        grouped = self.df.groupby(
            "turbine_id",
            sort=False
        )

        for turbine_id, group in grouped:

            group = (
                group
                .sort_values("timestamp")
                .reset_index(drop=True)
            )

            features = group[self.feature_cols].values
            targets = group[self.target_col].values

            n = len(group)

            if n < self.sequence_length:
                continue

            for start_idx in range(
                0,
                n - self.sequence_length,
                self.stride
            ):

                end_idx = start_idx + self.sequence_length

                sequence_x = features[start_idx:end_idx]

                target_y = targets[end_idx - 1]

                self.X.append(sequence_x)
                self.y.append(target_y)

                total_sequences += 1

        logger.info(
            f"Built {total_sequences:,} sequences"
        )

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        x = torch.tensor(
            self.X[idx],
            dtype=torch.float32
        )

        if self.task == "classification":

            y = torch.tensor(
                self.y[idx],
                dtype=torch.long
            )

        else:

            y = torch.tensor(
                self.y[idx],
                dtype=torch.float32
            )

        return x, y


# ─────────────────────────────────────────────────────
# DataLoader Helper
# ─────────────────────────────────────────────────────

def create_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0
):

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return loader


# ─────────────────────────────────────────────────────
# Dataset Loader
# ─────────────────────────────────────────────────────

def load_processed_data(
    data_dir: str = "data/processed"
):

    logger.info("Loading processed datasets")

    data_path = Path(data_dir)

    train_df = pd.read_parquet(
        data_path / "train.parquet"
    )

    val_df = pd.read_parquet(
        data_path / "val.parquet"
    )

    test_df = pd.read_parquet(
        data_path / "test.parquet"
    )

    feature_cols = joblib.load(
        data_path / "feature_columns.joblib"
    )

    logger.info(
        f"Train: {len(train_df):,} | "
        f"Val: {len(val_df):,} | "
        f"Test: {len(test_df):,}"
    )

    logger.info(
        f"Feature columns: {len(feature_cols)}"
    )

    return (
        train_df,
        val_df,
        test_df,
        feature_cols
    )


# ─────────────────────────────────────────────────────
# Create Datasets
# ─────────────────────────────────────────────────────

def create_datasets(
    data_dir: str = "data/processed",
    sequence_length: int = 60,
    stride: int = 5,
    task: str = "classification"
):

    (
        train_df,
        val_df,
        test_df,
        feature_cols
    ) = load_processed_data(data_dir)

    logger.info(
        f"Creating datasets | "
        f"Sequence length: {sequence_length}"
    )

    train_dataset = TurbineSequenceDataset(
        dataframe=train_df,
        feature_cols=feature_cols,
        target_col="fault_label",
        sequence_length=sequence_length,
        stride=stride,
        task=task
    )

    val_dataset = TurbineSequenceDataset(
        dataframe=val_df,
        feature_cols=feature_cols,
        target_col="fault_label",
        sequence_length=sequence_length,
        stride=stride,
        task=task
    )

    test_dataset = TurbineSequenceDataset(
        dataframe=test_df,
        feature_cols=feature_cols,
        target_col="fault_label",
        sequence_length=sequence_length,
        stride=stride,
        task=task
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset
    )


# ─────────────────────────────────────────────────────
# Create DataLoaders
# ─────────────────────────────────────────────────────

def create_dataloaders(
    data_dir: str = "data/processed",
    sequence_length: int = 60,
    stride: int = 5,
    batch_size: int = 64,
    task: str = "classification"
):

    (
        train_dataset,
        val_dataset,
        test_dataset
    ) = create_datasets(
        data_dir=data_dir,
        sequence_length=sequence_length,
        stride=stride,
        task=task
    )

    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = create_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = create_dataloader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    logger.info(
        f"Train batches: {len(train_loader):,}"
    )

    logger.info(
        f"Validation batches: {len(val_loader):,}"
    )

    logger.info(
        f"Test batches: {len(test_loader):,}"
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )


# ─────────────────────────────────────────────────────
# Example Usage
# ─────────────────────────────────────────────────────

def test_dataset_pipeline():

    logger.info("=" * 60)
    logger.info("Testing Dataset Pipeline")
    logger.info("=" * 60)

    (
        train_loader,
        val_loader,
        test_loader
    ) = create_dataloaders(
        sequence_length=60,
        stride=10,
        batch_size=32
    )

    logger.info("Checking first batch")

    for batch_x, batch_y in train_loader:

        logger.info(
            f"Input shape: {batch_x.shape}"
        )

        logger.info(
            f"Target shape: {batch_y.shape}"
        )

        logger.info(
            f"Example labels: {batch_y[:10]}"
        )

        break

    logger.info("=" * 60)
    logger.info("Dataset pipeline working correctly")
    logger.info("=" * 60)


# ─────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────

if __name__ == "__main__":

    test_dataset_pipeline()