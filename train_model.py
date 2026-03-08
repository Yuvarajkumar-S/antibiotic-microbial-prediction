import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from feature_extraction import fasta_to_features

# Extract features from genome dataset
features = fasta_to_features("bacteria.10.2.genomic.fna")

# Load labels
labels = pd.read_csv("labels.csv")

# Merge
data = pd.merge(features, labels, on="sample")

X = data.drop(["sample","resistance"], axis=1)
y = data["resistance"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "rf_model.pkl")

# Save feature names
joblib.dump(X.columns.tolist(), "feature_names.pkl")

print("Model and feature names saved")