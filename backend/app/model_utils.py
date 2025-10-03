import json
import joblib
from pathlib import Path
import numpy as np
import tensorflow as tf
from .config import MODELS_DIR

# paths
MODEL_PATH = MODELS_DIR / 'keras_house_model'
PREPROCESSOR_PATH = MODELS_DIR / 'preprocessor.joblib'
FEATURE_SPEC_PATH = MODELS_DIR / 'feature_spec.joblib'
FEATURE_COLUMNS_JSON = MODELS_DIR / 'feature_columns.json'

# load artifacts on import
_model = None
_preprocessor = None
_feature_spec = None
_feature_columns = None

def load_artifacts():
    global _model, _preprocessor, _feature_spec, _feature_columns
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run training first.")
        _model = tf.keras.models.load_model(str(MODEL_PATH))
    if _preprocessor is None:
        if not PREPROCESSOR_PATH.exists():
            raise FileNotFoundError(f"Preprocessor not found at {PREPROCESSOR_PATH}. Run training first.")
        _preprocessor = joblib.load(str(PREPROCESSOR_PATH))
    if _feature_spec is None:
        if not FEATURE_SPEC_PATH.exists():
            raise FileNotFoundError(f"Feature spec not found at {FEATURE_SPEC_PATH}.")
        _feature_spec = joblib.load(str(FEATURE_SPEC_PATH))
    if _feature_columns is None:
        if not FEATURE_COLUMNS_JSON.exists():
            raise FileNotFoundError(f"Feature columns file not found at {FEATURE_COLUMNS_JSON}.")
        with open(FEATURE_COLUMNS_JSON, 'r') as f:
            _feature_columns = json.load(f)
    return _model, _preprocessor, _feature_spec, _feature_columns

def preprocess_input(input_dict):
    """
    Expects a dict with keys matching the original feature names (numeric + categorical).
    Returns an array shaped (1, n_features_transformed)
    """
    _, preprocessor, feature_spec, _ = load_artifacts()
    numeric = feature_spec['numeric']
    categorical = feature_spec['categorical']

    # Build a single-row DataFrame-like 2D array in proper column order
    # For simplicity, we'll create a pandas DataFrame
    import pandas as pd
    row = {}
    # Fill numeric (if missing, set to 0)
    for n in numeric:
        val = input_dict.get(n)
        if val is None:
            # fallback: default to 0 (or could use precomputed medians if saved)
            val = 0
        row[n] = val
    for c in categorical:
        val = input_dict.get(c)
        if val is None:
            val = 'Unknown'
        row[c] = val
    df = pd.DataFrame([row])
    transformed = preprocessor.transform(df)
    return transformed

def predict(input_dict):
    model, _, _, feature_columns = load_artifacts()
    X = preprocess_input(input_dict)
    preds = model.predict(X)
    # model outputs raw number; ensure scalar
    return float(preds.flatten()[0])
