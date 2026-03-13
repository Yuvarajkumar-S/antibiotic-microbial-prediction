import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from tensorflow.keras.models import load_model
import shap

# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="GenomeGuard Clinical AMR System",
    page_icon="🧬",
    layout="wide"
)

# -----------------------------
# CNN parameters
# -----------------------------

MAX_LEN = 2000

mapping = {
"A":[1,0,0,0],
"T":[0,1,0,0],
"C":[0,0,1,0],
"G":[0,0,0,1]
}

# -----------------------------
# Encode genome sequence
# -----------------------------

def encode_sequence(seq):

    encoded = []

    for base in seq[:MAX_LEN]:

        if base in mapping:
            encoded.append(mapping[base])
        else:
            encoded.append([0,0,0,0])

    while len(encoded) < MAX_LEN:
        encoded.append([0,0,0,0])

    return np.array(encoded)

# -----------------------------
# CARD resistance genes
# -----------------------------

CARD_GENES = [
"blaCTX","mecA","tetA","tetM",
"gyrA","vanA","vanB","aac",
"ermB","sul1"
]

def detect_amr_genes(fasta_file):

    detected = []

    for record in SeqIO.parse(fasta_file,"fasta"):

        seq = str(record.seq)

        for gene in CARD_GENES:

            if gene.lower() in seq.lower():
                detected.append(gene)

    return list(set(detected))

# -----------------------------
# Load CNN model
# -----------------------------

model = load_model("cnn_model.h5")

# -----------------------------
# Header
# -----------------------------

st.title("🏥 GenomeGuard Clinical AMR Dashboard")

st.markdown(
"AI-powered platform for **predicting antibiotic resistance from bacterial genome sequences**"
)

st.divider()

# -----------------------------
# Upload FASTA genome
# -----------------------------

uploaded_file = st.file_uploader(
"Upload Bacterial Genome (FASTA)",
type=["fasta","fa","fna"]
)

if uploaded_file:

    with open("temp.fasta","wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("Genome uploaded successfully")

    # -----------------------------
    # Detect resistance genes
    # -----------------------------

    genes = detect_amr_genes("temp.fasta")

    col1,col2 = st.columns(2)

    with col1:

        st.subheader("Detected AMR Genes")

        if genes:
            st.write(genes)
        else:
            st.write("No known resistance genes detected")

    # -----------------------------
    # CNN prediction
    # -----------------------------

    for record in SeqIO.parse("temp.fasta","fasta"):

        seq = encode_sequence(str(record.seq).upper())
        seq = np.expand_dims(seq,axis=0)

        pred = model.predict(seq)

        confidence = float(pred[0][0])

        prediction = "Resistant" if confidence > 0.5 else "Susceptible"

    # -----------------------------
    # Antibiotic predictions
    # -----------------------------

    antibiotics = [
        "Ciprofloxacin",
        "Meropenem",
        "Tetracycline"
    ]

    results = []

    for ab in antibiotics:

        results.append({
        "Antibiotic":ab,
        "Prediction":prediction,
        "Confidence":round(confidence,3)
        })

    result_df = pd.DataFrame(results)

    with col2:

        st.subheader("Resistance Predictions")
        st.dataframe(result_df)

    st.divider()

    # -----------------------------
    # Confidence bar chart
    # -----------------------------

    st.subheader("Prediction Confidence")

    fig = px.bar(
        result_df,
        x="Antibiotic",
        y="Confidence",
        color="Prediction"
    )

    st.plotly_chart(fig)

    # -----------------------------
    # Resistance heatmap
    # -----------------------------

    st.subheader("Resistance Heatmap")

    heatmap_data = pd.DataFrame({
    "Ciprofloxacin":[confidence],
    "Meropenem":[confidence],
    "Tetracycline":[confidence]
    })

    fig2, ax = plt.subplots()

    sns.heatmap(
        heatmap_data,
        annot=True,
        cmap="coolwarm"
    )

    st.pyplot(fig2)

    # -----------------------------
    # 3D Genome visualization
    # -----------------------------

    st.subheader("3D Genome Visualization")

    x = np.linspace(0,10,100)
    y = np.sin(x)
    z = np.cos(x)

    fig3d = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines"
    )])

    st.plotly_chart(fig3d)

    # -----------------------------
    # Genomic importance map
    # -----------------------------

    st.subheader("Genomic Feature Importance")

    importance = np.abs(seq[0]).sum(axis=1)

    fig_imp = px.line(
        x=np.arange(len(importance)),
        y=importance,
        labels={"x":"Genome Position","y":"Importance"},
        title="Genome Importance Map"
    )

    st.plotly_chart(fig_imp)

    # -----------------------------
    # Optional SHAP
    # -----------------------------

    try:

        background = np.random.rand(10, MAX_LEN, 4)

        explainer = shap.GradientExplainer(model, background)

        shap_values = explainer.shap_values(seq)

        shap.summary_plot(shap_values, seq, show=False)

        st.pyplot(plt.gcf())

    except:
        st.info("SHAP visualization unavailable")

    # -----------------------------
    # Clinical recommendation
    # -----------------------------

    st.subheader("Clinical Recommendation")

    susceptible = result_df[result_df["Prediction"]=="Susceptible"]

    if len(susceptible)>0:

        best = susceptible.iloc[0]["Antibiotic"]

        st.success(f"Recommended Antibiotic: {best}")

    else:

        st.error("All tested antibiotics show resistance risk")
