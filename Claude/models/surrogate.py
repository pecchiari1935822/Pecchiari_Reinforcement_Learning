"""
models/surrogate.py
===================
Carica e gestisce il modello surrogato (rete neurale addestrata).
Questo modulo è critico: centralizza tutto il codice di loading/inference.
"""

import os
import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path
from utils.logger import logger
from config.paths import SURROGATE_MODEL_PATH, SCALERS_PATH


class SurrogateModel:
    """
    Wrapper per il modello surrogato (rete neurale).

    Responsabilità:
      1. Caricamento modello Keras
      2. Caricamento scaler (normalizzazione input/output)
      3. Inferenza veloce con tf.function
      4. Gestione errori e validazione

    Uso:
      >>> surrogate = SurrogateModel()
      >>> dof_values = np.array([0.1, 5.0, -65.0, 0.3, 0.5, 0.0, 0.0])
      >>> of_values = surrogate.predict(dof_values)  # array di 15 valori
    """

    def __init__(self, model_path: str = None, scaler_path: str = None):
        """
        Inizializza il surrogate model.

        Args:
            model_path: Path al file .keras (default: config)
            scaler_path: Path al file scalers.joblib (default: config)
        """
        self.model_path = model_path or str(SURROGATE_MODEL_PATH)
        self.scaler_path = scaler_path or str(SCALERS_PATH)

        # Valida file
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scalers not found: {self.scaler_path}")

        logger.info(f"Loading surrogate model from: {self.model_path}")

        # Carica modello
        self.keras_model = tf.keras.models.load_model(self.model_path)
        logger.info(f"Model loaded. Input shape: {self.keras_model.input_shape}")

        # Carica scaler
        logger.info(f"Loading scalers from: {self.scaler_path}")
        scalers = joblib.load(self.scaler_path)
        self.scaler_X = scalers['scaler_X']
        self.scaler_y = scalers['scaler_y']
        logger.info(f"Scaler types: {type(self.scaler_X).__name__}, {type(self.scaler_y).__name__}")

        # Estrai parametri scaler in numpy per velocità
        self._setup_scaling_functions()

        # Warm-up tf.function (compilazione grafo)
        logger.info("Warming up tf.function (first inference)...")
        self._warmup()
        logger.info("✅ Surrogate model ready!")

    def _setup_scaling_functions(self):
        """Crea funzioni veloci per scaling/unscaling."""
        scaler_type = type(self.scaler_X).__name__

        if scaler_type == "StandardScaler":
            self.X_offset = self.scaler_X.mean_.astype(np.float32)
            self.X_scale = self.scaler_X.scale_.astype(np.float32)
            self._scale_X = lambda x: (x - self.X_offset) / (self.X_scale + 1e-8)

        elif scaler_type == "MinMaxScaler":
            self.X_offset = self.scaler_X.data_min_.astype(np.float32)
            self.X_scale = self.scaler_X.data_range_.astype(np.float32)
            self._scale_X = lambda x: (x - self.X_offset) / (self.X_scale + 1e-8)

        else:
            logger.warning(f"Unknown scaler type: {scaler_type}. Fallback to sklearn.")
            self._scale_X = lambda x: self.scaler_X.transform(x.reshape(1, -1))[0].astype(np.float32)

        # Inverse scaling per output
        scaler_y_type = type(self.scaler_y).__name__

        if scaler_y_type == "StandardScaler":
            self.y_offset = self.scaler_y.mean_.astype(np.float32)
            self.y_scale = self.scaler_y.scale_.astype(np.float32)
            self._inverse_scale_y = lambda y: y * self.y_scale + self.y_offset

        elif scaler_y_type == "MinMaxScaler":
            self.y_offset = self.scaler_y.data_min_.astype(np.float32)
            self.y_scale = self.scaler_y.data_range_.astype(np.float32)
            self._inverse_scale_y = lambda y: y * self.y_scale + self.y_offset

        else:
            self._inverse_scale_y = lambda y: self.scaler_y.inverse_transform(
                y.reshape(1, -1)
            )[0].astype(np.float32)

    def _get_inference_fn(self):
        """Crea una tf.function compilata per inferenza veloce."""

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[1, self.keras_model.input_shape[-1]], dtype=tf.float32)
        ])
        def fast_infer(x):
            return self.keras_model(x, training=False)

        return fast_infer

    def _warmup(self):
        """Warm-up della tf.function (compilazione grafo TensorFlow)."""
        try:
            dummy = np.zeros(self.keras_model.input_shape[-1], dtype=np.float32)
            self.predict(dummy)
        except Exception as e:
            logger.warning(f"Warmup failed (non-critical): {e}")

    def predict(self, dof_values: np.ndarray) -> np.ndarray:
        """
        Effettua inferenza: DOF normalizzati → OF denormalizzati.

        Args:
            dof_values: array shape (7,) - DOF in unità fisiche reali

        Returns:
            array shape (15,) - OF in unità fisiche reali

        Raises:
            ValueError: Se input non valido
        """
        # Validazione
        if dof_values.shape != (7,):
            raise ValueError(f"Expected shape (7,), got {dof_values.shape}")

        dof_values = dof_values.astype(np.float32)

        # 1. Normalizza input
        x_scaled = self._scale_X(dof_values)
        x_tensor = x_scaled.reshape(1, -1).astype(np.float32)

        # 2. Inferenza veloce
        of_scaled = self.keras_model(x_tensor, training=False).numpy()

        # 3. Denormalizza output
        of_real = self._inverse_scale_y(of_scaled[0])

        # 4. Validazione output
        if np.isnan(of_real).any() or np.isinf(of_real).any():
            logger.warning(f"Invalid output from surrogate: {of_real}")
            raise ValueError("Surrogate returned NaN/Inf values")

        return of_real.astype(np.float32)

    def predict_batch(self, dof_batch: np.ndarray) -> np.ndarray:
        """
        Inferenza batch (più veloce di loop di predict).

        Args:
            dof_batch: array shape (N, 7)

        Returns:
            array shape (N, 15)
        """
        N = dof_batch.shape[0]
        dof_batch = dof_batch.astype(np.float32)

        # Normalizza
        x_scaled = np.array([self._scale_X(dof_batch[i]) for i in range(N)], dtype=np.float32)

        # Inferenza batch
        of_scaled = self.keras_model(x_scaled, training=False).numpy()

        # Denormalizza
        of_real = np.array([self._inverse_scale_y(of_scaled[i]) for i in range(N)], dtype=np.float32)

        return of_real


# Singleton: istanza globale
_surrogate_instance = None


def get_surrogate(model_path: str = None, scaler_path: str = None) -> SurrogateModel:
    """
    Restituisce l'istanza singleton del surrogate model.
    Usa questo per evitare di caricare il modello mille volte.

    Uso:
      >>> from models.surrogate import get_surrogate
      >>> surrogate = get_surrogate()
      >>> result = surrogate.predict(dof_array)
    """
    global _surrogate_instance
    if _surrogate_instance is None:
        _surrogate_instance = SurrogateModel(model_path, scaler_path)
    return _surrogate_instance


if __name__ == "__main__":
    # Test
    surrogate = SurrogateModel()
    test_dof = np.array([0.1, 5.0, -65.0, 0.3, 0.5, 0.0, 0.0], dtype=np.float32)
    of_result = surrogate.predict(test_dof)
    print(f"Test DOF: {test_dof}")
    print(f"Test OF: {of_result}")