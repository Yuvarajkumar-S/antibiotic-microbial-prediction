import joblib
from feature_extraction import fasta_to_features


model = joblib.load("rf_model.pkl")


def predict_genome(fasta_file):

    features = fasta_to_features(fasta_file)

    X = features.drop("sample", axis=1)

    prediction = model.predict(X)

    probability = model.predict_proba(X)

    return features["sample"], prediction, probability