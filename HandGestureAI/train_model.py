
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# File paths
DATA_FILE = "dataset/gestures.csv"
MODEL_FILE = "models/gesture_model.pkl"

# Check dataset
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        "Dataset nahi mila. Pehle collect_data.py run karo."
    )

df = pd.read_csv(DATA_FILE)

# Remove empty rows
df = df.dropna()

if "label" not in df.columns:
    raise ValueError("Dataset mein 'label' column nahi mila.")

# Features and labels
X = df.drop(columns=["label"]).to_numpy(dtype=float)
y = df["label"].to_numpy()

print("\n===== DATASET INFORMATION =====")
print("Total samples:", len(df))
print("\nSamples per gesture:")
print(pd.Series(y).value_counts())

# Validate labels
if len(set(y)) < 2:
    raise ValueError(
        "Training ke liye kam se kam 2 different gestures chahiye."
    )

if pd.Series(y).value_counts().min() < 2:
    raise ValueError(
        "Har gesture ke kam se kam 2 samples hone chahiye."
    )

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Train model
print("\nTraining model...")
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n===== MODEL RESULTS =====")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_FILE)

print("\nModel successfully saved!")
print("Location:", MODEL_FILE)
print("Next step: Run predict_gesture.py")
