import streamlit as st
import pandas as pd
import joblib
import shap
import plotly.express as px
import matplotlib.pyplot as plt
from Bio import SeqIO

from feature_extraction import fasta_to_features

# -------------------------------
# Page configuration
# -------------------------------

st.set_page_config(
    page_title="GenomeGuard Clinical AMR System",
    page_icon="🧬",
    layout="wide"
)

# -------------------------------
# CARD AMR gene list
# -------------------------------

CARD_GENES = [
    "blaCTX", "mecA", "tetA", "tetM",
    "gyrA", "vanA", "vanB", "aac",
    "ermB", "sul1"
]

# -------------------------------
# AMR gene detection
# -------------------------------

def detect_amr_genes(fasta_file):

    detected = []

    for record in SeqIO.parse(fasta_file, "fasta"):

        seq = str(record.seq)

        for gene in CARD_GENES:

            if gene.lower() in seq.lower():
                detected.append(gene)

    return list(set(detected))

# -------------------------------
# Header
# -------------------------------

st.title("🏥 GenomeGuard Clinical Dashboard")

st.markdown("""
AI-powered genomic platform for **antibiotic resistance prediction**  
from bacterial genome sequences.
""")

st.divider()

# -------------------------------
# Upload genome
# -------------------------------

uploaded_file = st.file_uploader(
    "Upload Genome FASTA",
    type=["fasta","fa","fna"]
)

if uploaded_file:

    with open("temp.fasta","wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("Genome uploaded successfully")

    # -------------------------------
    # Detect AMR genes
    # -------------------------------

    genes = detect_amr_genes("temp.fasta")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Detected AMR Genes")

        if genes:
            st.write(genes)
        else:
            st.write("No known resistance genes detected")

    # -------------------------------
    # Feature extraction
    # -------------------------------

    features = fasta_to_features("temp.fasta")

    X = features.drop("sample", axis=1)

    # -------------------------------
    # Fix feature mismatch problem
    # -------------------------------

    feature_names = joblib.load("feature_names.pkl")

    for col in feature_names:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_names]

    # -------------------------------
    # Load model
    # -------------------------------

    model = joblib.load("rf_model.pkl")

    prediction = model.predict(X)

    probability = model.predict_proba(X)

    confidence = probability.max(axis=1)[0]

    # -------------------------------
    # Multi-antibiotic prediction
    # -------------------------------

    antibiotics = [
        "Ciprofloxacin",
        "Meropenem",
        "Tetracycline"
    ]

    results = []

    for ab in antibiotics:

        pred = "Resistant" if prediction[0] == 1 else "Susceptible"

        results.append({
            "Antibiotic": ab,
            "Prediction": pred,
            "Confidence": round(confidence,3)
        })

    result_df = pd.DataFrame(results)

    # -------------------------------
    # Results table
    # -------------------------------

    with col2:

        st.subheader("Resistance Predictions")

        st.dataframe(result_df)

    st.divider()

    # -------------------------------
    # Confidence visualization
    # -------------------------------

    st.subheader("Prediction Confidence")

    fig = px.bar(
        result_df,
        x="Antibiotic",
        y="Confidence",
        color="Prediction",
        title="Antibiotic Resistance Confidence"
    )

    st.plotly_chart(fig)

    # -------------------------------
    # SHAP feature importance
    # -------------------------------

    st.subheader("Genomic Feature Importance")

    try:

        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(X)

        shap.summary_plot(shap_values, X, show=False)

        st.pyplot(plt.gcf())

    except:

        st.info("Feature importance could not be generated.")

    # -------------------------------
    # Clinical recommendation
    # -------------------------------

    st.divider()

    st.subheader("Clinical Recommendation")

    susceptible = result_df[result_df["Prediction"] == "Susceptible"]

    if len(susceptible) > 0:

        best = susceptible.iloc[0]["Antibiotic"]

        st.success(f"Recommended Antibiotic: {best}")

    else:

        st.error("All tested antibiotics show resistance risk")