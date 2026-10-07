"""Train a configurable feed-forward neural network on tabular CSV data.

TensorFlow/Keras implementation. Workflow:

1. Parse command-line arguments.
2. Load a CSV dataset, encode features/target, and split into train/val/test (70/15/15).
3. Fit the scaler (and label encoder) on the training data only.
4. Train a feed-forward ANN with dropout, using model.fit() + Keras callbacks for
   per-epoch logging, best-checkpoint-on-val-loss saving, and early stopping.
5. Restore the best checkpoint and evaluate on the held-out test split.

Supports both classification and regression tasks. Train split is reproducible via
a fixed random seed.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ---------------------------------------------------------------------------
# Config / CLI parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse and validate the command-line configuration for training."""
    parser = argparse.ArgumentParser(
        description="Train a TensorFlow ANN on tabular classification or regression data."
    )

    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to the CSV file containing the dataset.")
    parser.add_argument("--target-column", type=str, required=True,
                        help="Column name of the target variable.")
    parser.add_argument("--task", type=str, choices=["classification", "regression"],
                        default="classification",
                        help="Use classification or regression mode.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum number of epochs to train.")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Mini-batch size used during training.")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Adam learning rate.")
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[128, 64, 32],
                        help="Hidden layer sizes, e.g. --hidden-sizes 128 64 32.")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate applied after each hidden layer.")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience in epochs based on validation loss.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed used for reproducible splitting and model init.")
    parser.add_argument("--output-dir", type=str, default="runs",
                        help="Directory where timestamped run artifacts are written.")
    parser.add_argument("--save-splits", action="store_true",
                        help="Save train/validation/test CSVs in the run directory.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logger(run_dir: Path) -> logging.Logger:
    """Create a logger that writes to both the console and a file."""
    logger = logging.getLogger("tf_ann_training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(run_dir / "train.log", mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


class KerasEpochLogger(tf.keras.callbacks.Callback):
    """Route Keras's per-epoch metrics through our logging.Logger."""

    def __init__(self, logger: logging.Logger, task: str):
        super().__init__()
        self.logger = logger
        self.task = task

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_loss = logs.get("loss", float("nan"))
        val_loss = logs.get("val_loss", float("nan"))
        if self.task == "classification":
            train_acc = logs.get("accuracy", float("nan"))
            val_acc = logs.get("val_accuracy", float("nan"))
            self.logger.info(
                "Epoch %d | train_loss=%.6f train_acc=%.4f | val_loss=%.6f val_acc=%.4f",
                epoch + 1, train_loss, train_acc, val_loss, val_acc,
            )
        else:
            self.logger.info(
                "Epoch %d | train_loss=%.6f | val_loss=%.6f",
                epoch + 1, train_loss, val_loss,
            )


# ---------------------------------------------------------------------------
# Data loading & splitting
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


@dataclass
class DataSplits:
    """Container for train/validation/test arrays and their split metadata."""
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    scaler: StandardScaler
    label_encoder: LabelEncoder | None


def load_and_split_data(
    dataset_path: str,
    target_column: str,
    task: str,
    seed: int,
    save_splits: bool = False,
    output_dir: Path | None = None,
) -> DataSplits:
    """Load the CSV and split it into 70/15/15 train/validation/test partitions.

    Split logic:
      1) 70% train, 30% temp
      2) 50/50 split of temp into validation and test
    This keeps the total proportions at 70/15/15 while using a fixed random seed.
    Classification tasks use stratified splitting to preserve class distribution.

    Feature columns that are non-numeric are one-hot encoded. Classification
    targets are label-encoded to integers (0..n_classes-1) BEFORE splitting, so
    train/val/test all share the same label mapping.
    """
    df = pd.read_csv(dataset_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found in '{dataset_path}'.")
    if df.empty:
        raise ValueError(f"Dataset '{dataset_path}' is empty.")

    feature_columns = [col for col in df.columns if col != target_column]
    if not feature_columns:
        raise ValueError(
            f"No feature columns remain after removing target column '{target_column}'."
        )

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    # One-hot encode any non-numeric feature columns.
    X = pd.get_dummies(X, drop_first=True)

    label_encoder = None
    if task == "classification":
        if y.nunique() < 2:
            raise ValueError("Classification requires at least two unique target labels.")
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y), index=y.index, name=target_column)

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, train_size=0.70, random_state=seed, stratify=y,
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp,
        )
    else:
        y = y.astype(np.float32)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, train_size=0.70, random_state=seed,
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=seed,
        )

    train_df = X_train.copy()
    train_df[target_column] = y_train.to_numpy()

    val_df = X_val.copy()
    val_df[target_column] = y_val.to_numpy()

    test_df = X_test.copy()
    test_df[target_column] = y_test.to_numpy()

    scaler = StandardScaler()
    X_train_arr = scaler.fit_transform(X_train).astype(np.float32)
    X_val_arr = scaler.transform(X_val).astype(np.float32)
    X_test_arr = scaler.transform(X_test).astype(np.float32)

    if save_splits and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(output_dir / "train_split.csv", index=False)
        val_df.to_csv(output_dir / "validation_split.csv", index=False)
        test_df.to_csv(output_dir / "test_split.csv", index=False)

    return DataSplits(
        X_train=X_train_arr,
        X_val=X_val_arr,
        X_test=X_test_arr,
        y_train=y_train.to_numpy(),
        y_val=y_val.to_numpy(),
        y_test=y_test.to_numpy(),
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        scaler=scaler,
        label_encoder=label_encoder,
    )


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------


def get_num_classes(task: str, y_train: np.ndarray) -> int | None:
    """Infer the number of output classes for classification tasks."""
    if task != "classification":
        return None
    return int(len(np.unique(y_train)))


def build_model(
    input_dim: int,
    hidden_sizes: Sequence[int],
    dropout: float,
    task: str,
    num_classes: int | None,
    learning_rate: float,
) -> tf.keras.Model:
    """Build and compile a configurable feed-forward neural network."""
    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    x = inputs
    for units in hidden_sizes:
        x = tf.keras.layers.Dense(units, activation="relu")(x)
        if dropout > 0.0:
            x = tf.keras.layers.Dropout(dropout)(x)

    if task == "classification":
        if num_classes == 2:
            outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)
            loss = "binary_crossentropy"
            metrics = ["accuracy"]
        else:
            outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="output")(x)
            loss = "sparse_categorical_crossentropy"
            metrics = ["accuracy"]
    else:
        outputs = tf.keras.layers.Dense(1, name="output")(x)
        loss = "mse"
        metrics = ["mae"]

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="tabular_ann")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics,
    )
    return model


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------


def main() -> None:
    """Coordinate the full ANN training workflow and final model evaluation."""
    args = parse_args()
    set_seed(args.seed)

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_dir)
    logger.info("Starting TensorFlow ANN training run")
    logger.info("Dataset path: %s", dataset_path)
    logger.info("Target column: %s", args.target_column)
    logger.info("Task: %s", args.task)
    logger.info("Run directory: %s", run_dir)

    splits = load_and_split_data(
        dataset_path=str(dataset_path),
        target_column=args.target_column,
        task=args.task,
        seed=args.seed,
        save_splits=args.save_splits,
        output_dir=run_dir,
    )

    logger.info(
        "Data split sizes: train=%d, val=%d, test=%d",
        len(splits.X_train), len(splits.X_val), len(splits.X_test),
    )

    split_info = {
        "task": args.task,
        "train_size": len(splits.X_train),
        "val_size": len(splits.X_val),
        "test_size": len(splits.X_test),
        "seed": args.seed,
        "target_column": args.target_column,
        "num_features": int(splits.X_train.shape[1]),
    }
    if splits.label_encoder is not None:
        split_info["class_labels"] = list(map(str, splits.label_encoder.classes_))
    with open(run_dir / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    num_classes = get_num_classes(args.task, splits.y_train)
    model = build_model(
        input_dim=splits.X_train.shape[1],
        hidden_sizes=args.hidden_sizes,
        dropout=args.dropout,
        task=args.task,
        num_classes=num_classes,
        learning_rate=args.learning_rate,
    )
    model.summary(print_fn=lambda line: logger.info(line))

    best_model_path = run_dir / "best_model.keras"

    callbacks = [
        # Save the best model whenever val_loss improves.
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=0,
        ),
        # Stop training if val_loss hasn't improved for `patience` epochs.
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        # Route Keras's per-epoch output into our logger/log file.
        KerasEpochLogger(logger, args.task),
        # Also dump a machine-readable per-epoch CSV of metrics.
        tf.keras.callbacks.CSVLogger(str(run_dir / "metrics.csv")),
    ]

    logger.info(
        "Training configuration: epochs=%d, batch_size=%d, lr=%.6f, hidden_sizes=%s, "
        "dropout=%.2f, patience=%d",
        args.epochs, args.batch_size, args.learning_rate,
        args.hidden_sizes, args.dropout, args.patience,
    )

    history = model.fit(
        splits.X_train, splits.y_train,
        validation_data=(splits.X_val, splits.y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=0,
    )

    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    best_val_loss = float(np.min(history.history["val_loss"]))
    logger.info("Training finished. Best val_loss=%.6f at epoch %d", best_val_loss, best_epoch)

    # Reload the best checkpoint from disk (belt-and-suspenders alongside
    # restore_best_weights=True in EarlyStopping) and evaluate on the test set.
    best_model = tf.keras.models.load_model(best_model_path)
    test_results = best_model.evaluate(splits.X_test, splits.y_test, verbose=0, return_dict=True)

    if args.task == "classification":
        logger.info(
            "Final test evaluation: loss=%.6f | accuracy=%.4f",
            test_results["loss"], test_results.get("accuracy", float("nan")),
        )
    else:
        logger.info(
            "Final test evaluation: loss=%.6f | mae=%.6f",
            test_results["loss"], test_results.get("mae", float("nan")),
        )

    logger.info("Best model saved to %s", best_model_path)
    logger.info("Artifacts saved in: %s", run_dir)

    print(f"\nCompleted. Outputs saved in: {run_dir}")
    print(f"Best model: {best_model_path}")
    print(f"Training log: {run_dir / 'train.log'}")
    print(f"Per-epoch metrics CSV: {run_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()