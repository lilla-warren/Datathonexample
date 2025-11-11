import streamlit as st
import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# -----------------------------
# STREAMLIT PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="AI Clinical Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
    <style>
        /* Global page style */
        body {
            background-color: #f7faff;
            color: #0a2540;
            font-family: 'Inter', sans-serif;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #e7f0fa !important;
            border-right: 1px solid #d6e2ef;
        }

        /* Headers and text */
        h1, h2, h3 {
            color: #004080 !important;
            font-weight: 600 !important;
        }
        p, label, span {
            color: #1a2b3c !important;
        }

        /* Buttons */
        div.stButton > button {
            background-color: #004080 !important;
            color: white !important;
            border-radius: 8px;
            padding: 0.6em 1.2em;
            border: none;
        }
        div.stButton > button:hover {
            background-color: #0059b3 !important;
            color: #fff !important;
        }

        /* Right-aligned layout fix */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 20%;
            padding-right: 10%;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# PAGE TITLE
# -----------------------------
st.title("🩺 AI Clinical Analytics Dashboard")

st.write("""
This platform allows clinicians and researchers to **upload datasets**, 
**train ML models**, and **analyze patient outcomes** using AI insights.  
All computations run securely and locally.
""")

# -----------------------------
# LAYOUT: RIGHT-ALIGNED CONTENT
# -----------------------------
col_left, col_right = st.columns([0.3, 0.7])

with col_right:
    st.subheader("📁 Upload Clinical Dataset")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
            st.write("### Data Preview")
            st.dataframe(df.head())

            # -----------------------------
            # TARGET SELECTION
            # -----------------------------
            target_col = st.selectbox("Select Target Variable (e.g., Diagnosis, Outcome)", df.columns)

            # -----------------------------
            # FEATURE SELECTION
            # -----------------------------
            feature_cols = st.multiselect(
                "Select Feature Columns (optional)",
                [col for col in df.columns if col != target_col],
                default=[col for col in df.columns if col != target_col]
            )

            # -----------------------------
            # MODEL TRAINING SECTION
            # -----------------------------
            st.subheader("⚙️ Model Configuration")
            test_size = st.slider("Test Split (%)", 10, 50, 20, step=5)
            model_type = st.selectbox("Select Model", ["Random Forest", "Logistic Regression"])

            if st.button("Run Clinical Analysis"):
                X = df[feature_cols]
                y = df[target_col]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size/100, random_state=42
                )

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model = RandomForestClassifier(random_state=42)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

                acc = accuracy_score(y_test, y_pred)
                st.metric("Model Accuracy", f"{acc*100:.2f}%")
                st.text("Classification Report:")
                st.code(classification_report(y_test, y_pred))

                # -----------------------------
                # SHAP EXPLAINABILITY
                # -----------------------------
                st.subheader("🔍 Model Explainability (SHAP)")
                explainer = shap.Explainer(model, X_train)
                shap_values = explainer(X_test)
                fig, ax = plt.subplots()
                shap.summary_plot(shap_values, X_test, show=False)
                st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

with col_left:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966488.png", width=120)
    st.markdown("### Navigation")
    st.write("- 📂 Upload Data\n- 🧠 Train Model\n- 📊 View Insights\n- 💡 Explain Results")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>© 2025 AI Clinical Analytics Dashboard — Designed for research and healthcare innovation.</p>",
    unsafe_allow_html=True
)
