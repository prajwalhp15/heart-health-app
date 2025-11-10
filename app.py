# 🫀 Heart Health Archetype Predictor using K-Means & SOM (Streamlit)
# Author: Prajwal HP (Final Updated Version with Cluster Visualization)

import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

# ===============================
# Load Trained Models
# ===============================
with open('kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('pca_model.pkl', 'rb') as f:
    pca = pickle.load(f)

# SOM is optional
try:
    with open('som_model.pkl', 'rb') as f:
        som = pickle.load(f)
except:
    som = None

# ===============================
# Streamlit Page Setup
# ===============================
st.set_page_config(page_title="Heart Health Predictor", layout="centered")
st.title("Heart Health Archetype Discovery")
st.markdown("Enter your clinical details below to predict your heart health cluster.")

# ===============================
# Input Form
# ===============================
with st.form("patient_form"):
    age = st.number_input("Age (years)", 20, 90, 45)
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 400, 200)
    thalach = st.number_input("Maximum Heart Rate Achieved (bpm)", 60, 220, 150)
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.5, 1.0, step=0.1)
    submit = st.form_submit_button("Predict Cluster")

# ===============================
# Cluster Meaning Detection
# ===============================
cluster_centers = kmeans.cluster_centers_

# Compute a “risk score” for each cluster: BP + Chol + Oldpeak - HR
cluster_scores = []
for center in cluster_centers:
    score = (center[1] + center[2] + center[4]) - center[3]
    cluster_scores.append(score)

# Rank clusters by severity
sorted_indices = np.argsort(cluster_scores)
healthy_cluster = sorted_indices[0]
atrisk_cluster = sorted_indices[1]
critical_cluster = sorted_indices[2]

cluster_map = {
    healthy_cluster: (" Healthy Group", "Balanced vitals and good heart rate.", "green"),
    atrisk_cluster: (" At-Risk Group", "Slightly elevated BP or cholesterol. Monitor regularly.", "orange"),
    critical_cluster: ("Critical Group", "High BP and cholesterol. Requires medical attention.", "red"),
}

# ===============================
# Prediction Logic
# ===============================
if submit:
    input_data = np.array([[age, trestbps, chol, thalach, oldpeak]])
    input_scaled = scaler.transform(input_data)
    cluster = kmeans.predict(input_scaled)[0]

    label, desc, color = cluster_map[cluster]

    # --- Display Output ---
    st.markdown("---")
    st.subheader(" Prediction Result:")
    st.markdown(f"### You belong to: **{label}**")
    st.markdown(f"**Interpretation:** {desc}")

    # --- Clinical Message ---
    if "Healthy" in label:
        st.success("Clinical Status: Normal cardiovascular profile. Maintain healthy habits.")
    elif "At-Risk" in label:
        st.warning("Clinical Status: Borderline hypertension or cholesterol. Recommend monitoring and lifestyle management.")
    else:
        st.error("Clinical Status: Possible hypertension or ischemic changes. Recommend clinical evaluation and treatment.")

    # ===============================
    # PCA Visualization
    # ===============================
    transformed = pca.transform(input_scaled)

    # --- Single input PCA point ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(transformed[0][0], transformed[0][1], color=color, s=200, marker='x')
    ax.set_title("PCA Projection of Your Heart Health Data")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    st.pyplot(fig)

    # --- Optional Enhanced Cluster Visualization ---
    st.markdown("###  Compare with all cluster centers")
    if st.checkbox("Show all clusters in PCA plot"):
        cluster_points = pca.transform(kmeans.cluster_centers_)

        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["green", "orange", "red"]
        labels = ["Healthy", "At-Risk", "Critical"]

        # Plot cluster centers
        for i, (x, y) in enumerate(cluster_points):
            ax.scatter(x, y, color=colors[i], label=f"{labels[i]} Cluster Center",
                       s=200, marker='o', edgecolors='black')

        # Plot user input
        ax.scatter(transformed[0][0], transformed[0][1], color=color, s=250,
                   marker='x', label='Your Data', linewidths=3)

        ax.set_title("PCA Projection with Cluster Centers")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.legend()
        st.pyplot(fig)

    # ===============================
    # Cluster Summary Table
    # ===============================
    st.markdown("###  Cluster Averages (Clinical Summary)")
    feature_names = ['Age', 'Resting BP', 'Cholesterol', 'Max HR', 'Oldpeak']
    cluster_summary = pd.DataFrame(cluster_centers, columns=feature_names)
    cluster_summary['Risk Level'] = ['Healthy', 'At-Risk', 'Critical']
    st.dataframe(cluster_summary.style.highlight_max(color="lightcoral").highlight_min(color="lightgreen"))

    st.success("Prediction completed successfully!")
