import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import tensorflow as tf
from tensorflow.keras import layers
import json
import os

# -----------------------------
# Argument Parser
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True, help="Path to CSV dataset")
args = parser.parse_args()

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(args.data)

# Drop ID and Date (not useful for prediction)
df = df.drop(columns=["id", "Date"], errors="ignore")

# Target column
y = df["Price"]
X = df.drop(columns=["Price"])

# -----------------------------
# Define feature types
# -----------------------------
numeric_features = [
    "number of bedrooms", "number of bathrooms", "living area", "lot area",
    "number of floors", "number of views", "condition of the house", "grade of the house",
    "Area of the house(excluding basement)", "Area of the basement", "Built Year",
    "Renovation Year", "Postal Code", "Lattitude", "Longitude",
    "living_area_renov", "lot_area_renov", "Number of schools nearby",
    "Distance from the airport"
]

categorical_features = [
    "waterfront present"  # yes/no
]

# -----------------------------
# Preprocessor
# -----------------------------
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# -----------------------------
# Split dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Preprocess features
# -----------------------------
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Get feature names after transformation
# Works for sklearn >=1.0
cat_cols = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features)
all_feature_names = np.concatenate([numeric_features, cat_cols])

# -----------------------------
# Save preprocessors + feature names
# -----------------------------
os.makedirs("../backend/app/models", exist_ok=True)

joblib.dump(preprocessor, "../backend/app/models/preprocessor.joblib")
joblib.dump(all_feature_names.tolist(), "../backend/app/models/feature_spec.joblib")

with open("../backend/app/models/feature_columns.json", "w") as f:
    json.dump(all_feature_names.tolist(), f)

# -----------------------------
# Build Neural Network
# -----------------------------
model = tf.keras.Sequential([
    layers.Input(shape=(X_train_processed.shape[1],)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="linear")
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# -----------------------------
# Train model
# -----------------------------
history = model.fit(
    X_train_processed, y_train,
    validation_data=(X_test_processed, y_test),
    epochs=20,
    batch_size=32,
    verbose=1
)

# -----------------------------
# Save model (Keras 3 compatible)
# -----------------------------
model.save("../backend/app/models/house_model.keras")

# -----------------------------
# Evaluate
# -----------------------------
# -----------------------------
# Evaluate
# -----------------------------
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Use processed test data!
y_pred = model.predict(X_test_processed)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

