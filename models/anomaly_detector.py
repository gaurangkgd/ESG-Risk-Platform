"""
Financial Anomaly Detector - TensorFlow Autoencoder
Detects anomalies in ESG-related financial metrics using an autoencoder.
High reconstruction error = unusual/anomalous pattern = potential risk signal.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, List, Tuple, Optional
import json


class ESGAnomalyDetector:
    """
    TensorFlow-based autoencoder for detecting anomalies in ESG financial metrics.
    
    Trained on 'normal' ESG metric patterns, it flags companies whose
    metrics deviate significantly (e.g., sudden emission spikes while
    claiming reduction targets).
    """

    # Standard ESG financial metrics we track
    METRIC_NAMES = [
        "carbon_intensity",          # tCO2e per $M revenue
        "energy_consumption_growth", # YoY % change
        "water_usage_intensity",     # m3 per unit production
        "waste_recycling_rate",      # % recycled
        "employee_turnover_rate",    # % annual
        "gender_pay_gap",            # % difference
        "board_independence_ratio",  # % independent directors
        "esg_disclosure_score",      # 0-100 completeness
        "controversies_count",       # # of ESG controversies
        "capex_green_ratio"          # % capex in green projects
    ]

    def __init__(self, input_dim: int = 10, encoding_dim: int = 4):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.model = self._build_autoencoder()
        self.threshold = None
        self._is_trained = False
        print("[ESGAnomalyDetector] TensorFlow autoencoder initialized")

    def _build_autoencoder(self) -> keras.Model:
        """Build encoder-decoder architecture."""
        # Encoder
        inputs = keras.Input(shape=(self.input_dim,), name="esg_metrics_input")
        encoded = keras.layers.Dense(8, activation="relu", name="encoder_1")(inputs)
        encoded = keras.layers.BatchNormalization(name="bn_1")(encoded)
        encoded = keras.layers.Dense(self.encoding_dim, activation="relu", name="bottleneck")(encoded)

        # Decoder
        decoded = keras.layers.Dense(8, activation="relu", name="decoder_1")(encoded)
        decoded = keras.layers.BatchNormalization(name="bn_2")(decoded)
        decoded = keras.layers.Dense(self.input_dim, activation="sigmoid", name="reconstruction")(decoded)

        model = keras.Model(inputs, decoded, name="esg_autoencoder")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"]
        )
        return model

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Min-max normalize to [0, 1]."""
        self._min = data.min(axis=0)
        self._max = data.max(axis=0)
        range_ = self._max - self._min
        range_[range_ == 0] = 1  # Avoid division by zero
        return (data - self._min) / range_

    def train(self, training_data: np.ndarray, epochs: int = 50, batch_size: int = 16):
        """
        Train the autoencoder on 'normal' ESG metric patterns.
        
        Args:
            training_data: Shape (n_samples, n_metrics) - normal company data
            epochs: Training epochs
            batch_size: Batch size
        """
        normalized = self._normalize(training_data)

        history = self.model.fit(
            normalized, normalized,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.15,
            shuffle=True,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ]
        )

        # Set anomaly threshold at 95th percentile of training reconstruction errors
        reconstructions = self.model.predict(normalized, verbose=0)
        errors = np.mean(np.power(normalized - reconstructions, 2), axis=1)
        self.threshold = float(np.percentile(errors, 95))
        self._is_trained = True

        print(f"[ESGAnomalyDetector] Trained for {len(history.history['loss'])} epochs")
        print(f"[ESGAnomalyDetector] Anomaly threshold set at: {self.threshold:.4f}")
        return history

    def generate_synthetic_training_data(self, n_samples: int = 500) -> np.ndarray:
        """
        Generate synthetic 'normal' ESG metric data for demo/training purposes.
        In production, replace with real benchmark data.
        """
        np.random.seed(42)
        data = np.column_stack([
            np.random.normal(150, 30, n_samples),    # carbon_intensity
            np.random.normal(-5, 3, n_samples),       # energy_consumption_growth (declining = good)
            np.random.normal(2.5, 0.5, n_samples),    # water_usage_intensity
            np.random.normal(75, 10, n_samples),      # waste_recycling_rate
            np.random.normal(12, 3, n_samples),       # employee_turnover_rate
            np.random.normal(8, 4, n_samples),        # gender_pay_gap
            np.random.normal(65, 10, n_samples),      # board_independence_ratio
            np.random.normal(72, 12, n_samples),      # esg_disclosure_score
            np.random.normal(2, 1, n_samples),        # controversies_count
            np.random.normal(18, 5, n_samples),       # capex_green_ratio
        ])
        return np.abs(data)  # Keep positive

    def predict_anomaly(self, metrics: np.ndarray) -> Dict:
        """
        Detect if a company's ESG metrics are anomalous.
        
        Args:
            metrics: Shape (1, n_metrics) or (n_metrics,) - company metrics
        
        Returns:
            Dict with anomaly score, is_anomaly flag, and per-metric breakdown
        """
        if not self._is_trained:
            # Auto-train on synthetic data if not trained
            print("[ESGAnomalyDetector] Auto-training on synthetic data...")
            synthetic = self.generate_synthetic_training_data()
            self.train(synthetic)

        metrics = np.array(metrics).reshape(1, -1)

        # Normalize using training stats
        range_ = self._max - self._min
        range_[range_ == 0] = 1
        normalized = (metrics - self._min) / range_
        normalized = np.clip(normalized, 0, 1)

        reconstruction = self.model.predict(normalized, verbose=0)
        reconstruction_error = float(np.mean(np.power(normalized - reconstruction, 2)))

        # Per-metric anomaly scores
        per_metric_errors = np.power(normalized - reconstruction, 2)[0]
        metric_anomalies = {
            name: {
                "value": float(metrics[0][i]),
                "anomaly_score": float(per_metric_errors[i]),
                "is_anomalous": float(per_metric_errors[i]) > (self.threshold / self.input_dim)
            }
            for i, name in enumerate(self.METRIC_NAMES[:self.input_dim])
        }

        anomaly_score_pct = min(100, (reconstruction_error / (self.threshold + 1e-8)) * 50)

        return {
            "reconstruction_error": reconstruction_error,
            "anomaly_threshold": self.threshold,
            "is_anomalous": reconstruction_error > self.threshold,
            "anomaly_score": round(anomaly_score_pct, 2),
            "risk_signal": (
                "ANOMALOUS - Potential data manipulation or greenwashing"
                if reconstruction_error > self.threshold
                else "NORMAL - Metrics within expected range"
            ),
            "per_metric_breakdown": metric_anomalies
        }

    def save_model(self, path: str = "./models/anomaly_detector"):
        """Save TensorFlow model."""
        self.model.save(path)
        print(f"[ESGAnomalyDetector] Model saved to {path}")

    def load_model(self, path: str = "./models/anomaly_detector"):
        """Load saved TensorFlow model."""
        self.model = keras.models.load_model(path)
        self._is_trained = True
        print(f"[ESGAnomalyDetector] Model loaded from {path}")
