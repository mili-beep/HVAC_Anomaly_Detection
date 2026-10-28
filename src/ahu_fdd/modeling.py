"""Shared leakage-safe model adapters used by detection and diagnosis runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

MODEL_NAMES = (
    "normal_range_rule",
    "isolation_forest",
    "xgboost",
    "lstm_autoencoder",
)


def load_detection_config(path: Path) -> dict[str, Any]:
    """Load and validate the shared binary-detection configuration."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The detection configuration must be a mapping")
    if str(config.get("normal_label", "")).strip() == "":
        raise ValueError("The detection configuration must define normal_label")
    for model in MODEL_NAMES:
        if model not in config.get("models", {}):
            raise ValueError(f"The detection configuration must define {model}")
    return config


def majority_prediction(y_train: pd.Series, count: int) -> np.ndarray:
    """Return the training-fold majority class for a degenerate model case."""
    majority = int(y_train.value_counts().sort_index().idxmax())
    return np.full(count, majority, dtype=int)


class FoldTransformer:
    """Fit imputation/scaling on training rows only and transform another fold."""

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self.usable: list[str] = []
        self.pipeline: Any = None

    def fit(self, train: pd.DataFrame) -> "FoldTransformer":
        finite = train[self.columns].replace([np.inf, -np.inf], np.nan)
        self.usable = [
            column
            for column in self.columns
            if finite[column].notna().any() and finite[column].nunique(dropna=True) > 1
        ]
        if self.usable:
            self.pipeline = make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler()
            )
            self.pipeline.fit(finite[self.usable])
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.usable or self.pipeline is None:
            return np.empty((len(frame), 0), dtype=float)
        values = frame[self.usable].replace([np.inf, -np.inf], np.nan)
        return np.asarray(self.pipeline.transform(values), dtype=float)


def isolation_forest(
    train_matrix: np.ndarray,
    test_matrix: np.ndarray,
    y_train: pd.Series,
    config: dict[str, Any],
) -> np.ndarray:
    """Fit the unsupervised detector on normal training rows only."""
    normal = train_matrix[y_train.to_numpy() == 0]
    if len(normal) < 2 or train_matrix.shape[1] == 0:
        return majority_prediction(y_train, len(test_matrix))
    model_config = config["models"]["isolation_forest"]
    configured_max_samples = model_config.get("max_samples", "auto")
    max_samples = (
        min(int(configured_max_samples), len(normal))
        if configured_max_samples != "auto"
        else "auto"
    )
    model = IsolationForest(
        n_estimators=int(model_config["n_estimators"]),
        max_samples=max_samples,
        random_state=int(model_config["random_state"]),
        n_jobs=int(model_config.get("n_jobs", 2)),
    )
    model.fit(normal)
    threshold = np.quantile(
        model.score_samples(normal), float(model_config["normal_score_quantile"])
    )
    return (model.score_samples(test_matrix) <= threshold).astype(int)


def xgboost(
    train_matrix: np.ndarray,
    test_matrix: np.ndarray,
    y_train: pd.Series,
    config: dict[str, Any],
) -> np.ndarray:
    """Fit the configured supervised XGBoost binary detector."""
    if train_matrix.shape[1] == 0 or y_train.nunique() < 2:
        return majority_prediction(y_train, len(test_matrix))
    model_config = dict(config["models"]["xgboost"])
    normal = int((y_train == 0).sum())
    fault = int((y_train == 1).sum())
    model_config["scale_pos_weight"] = normal / fault if fault else 1.0
    model = XGBClassifier(**model_config)
    model.fit(train_matrix, y_train.to_numpy())
    return model.predict(test_matrix).astype(int)


def predict_binary_model(
    model_name: str,
    raw_train: pd.DataFrame,
    raw_eval: pd.DataFrame,
    feature_train: pd.DataFrame,
    feature_eval: pd.DataFrame,
    y_train: pd.Series,
    y_eval: pd.Series,
    columns: list[str],
    config: dict[str, Any],
) -> tuple[np.ndarray, pd.Series]:
    """Run one declared binary detector using the common fold-safe path.

    ``raw_*`` are used by the transparent normal-range rule, while all learned
    models use the generated feature frames and a transformer fitted on the
    training rows only.  The returned boolean series marks rows scored by the
    model (LSTM windows can leave leading rows unscored).
    """
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown detection model: {model_name}")
    if model_name == "normal_range_rule":
        from ahu_fdd.baselines import normal_range_prediction

        if set(columns).issubset(raw_train.columns):
            rule_columns = columns
        else:
            rule_columns = [
                column.removeprefix("sensor__")
                for column in columns
                if column.startswith("sensor__")
                and column.removeprefix("sensor__") in raw_train.columns
            ]
        predictions = normal_range_prediction(
            raw_train,
            raw_eval,
            y_train,
            rule_columns,
            config["normal_range_rule"],
        )
        return predictions, pd.Series(True, index=raw_eval.index)
    if model_name == "lstm_autoencoder":
        sequence_train = feature_train.copy()
        sequence_eval = feature_eval.copy()
        if "row_id" not in sequence_train:
            sequence_train["row_id"] = sequence_train.index.to_numpy()
        if "row_id" not in sequence_eval:
            sequence_eval["row_id"] = sequence_eval.index.to_numpy()
        return lstm_autoencoder(
            sequence_train,
            sequence_eval,
            y_train,
            y_eval,
            columns,
            config,
        )
    transformer = FoldTransformer(columns).fit(feature_train)
    train_matrix = transformer.transform(feature_train)
    eval_matrix = transformer.transform(feature_eval)
    if model_name == "isolation_forest":
        predictions = isolation_forest(train_matrix, eval_matrix, y_train, config)
    else:
        predictions = xgboost(train_matrix, eval_matrix, y_train, config)
    return predictions, pd.Series(True, index=feature_eval.index)


def lstm_available() -> bool:
    """Return whether the optional TensorFlow dependency is installed."""
    try:
        import tensorflow  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def sequence_windows(
    frame: pd.DataFrame,
    matrix: np.ndarray,
    sequence_length: int,
    max_gap_hours: float,
) -> tuple[np.ndarray, list[int]]:
    """Create within-AHU windows without crossing a timestamp gap."""
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    if max_gap_hours <= 0:
        raise ValueError("max_gap_hours must be positive")
    chunks: list[np.ndarray] = []
    rows: list[int] = []
    for _, indices in frame.groupby(["building", "ahu"], sort=False).groups.items():
        ordered = frame.loc[indices].sort_values(["timestamp", "row_id"], kind="stable")
        positions = frame.index.get_indexer(ordered.index)
        timestamps = pd.to_datetime(ordered["timestamp"], errors="coerce")
        if timestamps.isna().any():
            raise ValueError("Sequence input contains unparseable timestamps")
        hours = timestamps.diff().dt.total_seconds().div(3600)
        duplicate = frame.loc[ordered.index].get(
            "duplicate_ahu_timestamp", pd.Series(False, index=ordered.index)
        ).astype(bool)
        boundary = hours.gt(max_gap_hours) | hours.le(0) | duplicate | duplicate.shift(
            1, fill_value=False
        )
        breaks = np.flatnonzero(boundary.to_numpy()[1:]) + 1
        starts = np.concatenate(([0], breaks))
        ends = np.concatenate((breaks, [len(positions)]))
        for start, end in zip(starts, ends):
            segment = positions[start:end]
            for offset in range(sequence_length - 1, len(segment)):
                window_positions = segment[offset - sequence_length + 1 : offset + 1]
                chunks.append(matrix[window_positions])
                rows.append(int(segment[offset]))
    if not chunks:
        return np.empty((0, sequence_length, matrix.shape[1]), dtype=float), []
    return np.stack(chunks), rows


def lstm_autoencoder(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    columns: list[str],
    config: dict[str, Any],
) -> tuple[np.ndarray, pd.Series]:
    """Train an LSTM autoencoder on normal training sequences only."""
    if not lstm_available():
        raise RuntimeError(
            "lstm_autoencoder requires TensorFlow; install the optional deep extra"
        )
    import tensorflow as tf

    transformer = FoldTransformer(columns).fit(train.loc[y_train.eq(0)])
    if not transformer.usable:
        return majority_prediction(y_train, len(test)), pd.Series(True, index=test.index)
    sequence_length = int(config["models"]["lstm_autoencoder"]["sequence_length"])
    train_matrix = transformer.transform(train)
    test_matrix = transformer.transform(test)

    sequence_gap = float(
        config["models"]["lstm_autoencoder"].get("sequence_max_gap_hours", 1)
    )
    train_sequences, _ = sequence_windows(
        train, train_matrix, sequence_length, sequence_gap
    )
    test_sequences, test_rows = sequence_windows(
        test, test_matrix, sequence_length, sequence_gap
    )
    if len(train_sequences) == 0 or len(test_sequences) == 0:
        return majority_prediction(y_train, len(test)), pd.Series(False, index=test.index)
    maximum = int(
        config["models"]["lstm_autoencoder"].get("max_train_sequences", 20000)
    )
    if len(train_sequences) > maximum:
        rng = np.random.default_rng(
            int(config["models"]["lstm_autoencoder"]["random_state"])
        )
        selected = rng.choice(len(train_sequences), size=maximum, replace=False)
        train_sequences = train_sequences[selected]
    hidden = int(config["models"]["lstm_autoencoder"]["hidden_units"])
    inputs = tf.keras.Input(shape=(sequence_length, train_sequences.shape[2]))
    encoded = tf.keras.layers.LSTM(hidden)(inputs)
    decoded = tf.keras.layers.RepeatVector(sequence_length)(encoded)
    decoded = tf.keras.layers.LSTM(hidden, return_sequences=True)(decoded)
    decoded = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(train_sequences.shape[2])
    )(decoded)
    model = tf.keras.Model(inputs, decoded)
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        train_sequences,
        train_sequences,
        epochs=int(config["models"]["lstm_autoencoder"]["epochs"]),
        batch_size=int(config["models"]["lstm_autoencoder"]["batch_size"]),
        verbose=0,
        shuffle=False,
    )
    train_errors = np.mean(
        np.square(model.predict(train_sequences, verbose=0) - train_sequences), axis=(1, 2)
    )
    test_errors = np.mean(
        np.square(model.predict(test_sequences, verbose=0) - test_sequences), axis=(1, 2)
    )
    threshold = np.quantile(
        train_errors,
        float(config["models"]["lstm_autoencoder"]["normal_error_quantile"]),
    )
    predictions = majority_prediction(y_train, len(test))
    predictions[np.asarray(test_rows)] = (test_errors >= threshold).astype(int)
    scored = pd.Series(False, index=test.index)
    scored.iloc[test_rows] = True
    return predictions, scored
