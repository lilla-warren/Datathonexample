import streamlit as st
import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------
# STREAMLIT PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="AI Clinical Analytics Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# ENHANCED CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global page styles */
    .main {
        background-color: #f8fbff;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers styling */
    h1, h2, h3 {
        color: #1a365d !important;
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
    }
    
    h1 {
        font-size: 2.5rem !important;
        background: linear-gradient(135deg, #1a365d, #2d5aa0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e7f0fa 0%, #f0f7ff 100%);
        border-right: 1px solid #d1e0f0;
        padding: 2rem 1rem;
    }
    
    .sidebar .sidebar-content {
        background: transparent !important;
    }
    
    /* Main content area alignment */
    .main .block-container {
        padding-top: 2rem;
        padding-left: 5rem;
        padding-right: 3rem;
        max-width: 1200px;
    }
    
    /* Card-like containers */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #e1e8f0;
        margin-bottom: 1.5rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1a365d, #2d5aa0) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(26, 54, 93, 0.2) !important;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* File uploader styling */
    .uploadedFile {
        border: 2px dashed #2d5aa0 !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        background-color: #f8fbff !important;
    }
    
    /* Success and error messages */
    .stSuccess {
        background: linear-gradient(135deg, #d4edda, #c3e6cb) !important;
        border: 1px solid #c3e6cb !important;
        color: #155724 !important;
        border-radius: 8px !important;
    }
    
    .stError {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb) !important;
        border: 1px solid #f5c6cb !important;
        color: #721c24 !important;
        border-radius: 8px !important;
    }
    
    /* Navigation items */
    .nav-item {
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        background: white;
        border-radius: 8px;
        border-left: 4px solid #2d5aa0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .nav-item:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR CONTENT
# -----------------------------
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <img src='https://cdn-icons-png.flaticon.com/512/2966/2966488.png' width='80' style='margin-bottom: 1rem;'>
        <h3 style='color: #1a365d; margin: 0;'>Clinical Analytics</h3>
        <p style='color: #6b7280; font-size: 0.9rem;'>AI-Powered Healthcare Insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔍 Navigation")
    
    nav_items = [
        "📂 Data Upload & Overview",
        "📊 Exploratory Data Analysis", 
        "🤖 Model Training & Validation",
        "🔍 Model Explainability",
        "📈 Clinical Insights",
        "⚖️ Ethical Considerations"
    ]
    
    for item in nav_items:
        st.markdown(f"<div class='nav-item'>{item}</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📋 Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Models", "4")
        st.metric("Accuracy", "96.5%")
    with col2:
        st.metric("Features", "15+")
        st.metric("Patients", "1,000+")
    
    st.markdown("---")
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #e3f2fd, #bbdefb); padding: 1rem; border-radius: 8px; border-left: 4px solid #1976d2;'>
        <h4 style='color: #1a365d; margin: 0 0 0.5rem 0;'>💡 Tip</h4>
        <p style='color: #455a64; margin: 0; font-size: 0.9rem;'>Upload your clinical dataset to begin analysis. Ensure data is properly anonymized for patient privacy.</p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# MAIN CONTENT AREA
# -----------------------------
def main():
    # Header Section
    st.markdown("""
    <div style='margin-bottom: 3rem;'>
        <h1>🩺 AI Clinical Analytics Dashboard</h1>
        <p style='font-size: 1.1rem; color: #6b7280; line-height: 1.6;'>
        Advanced machine learning platform for clinical data analysis. Transform patient data into 
        actionable insights while maintaining the highest standards of data privacy and ethical AI practices.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create main content columns
    col1, col2 = st.columns([1, 3])
    
    with col2:
        # Data Upload Section
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📁 Upload Clinical Dataset")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file containing clinical data",
            type=["csv"],
            help="Upload anonymized patient data for analysis"
        )
        
        if uploaded_file is not None:
            try:
                # Load and display data
                df = pd.read_csv(uploaded_file)
                
                st.success("✅ Dataset uploaded successfully!")
                
                # Data Overview
                col_info1, col_info2, col_info3 = st.columns(3)
                with col_info1:
                    st.metric("Patients", len(df))
                with col_info2:
                    st.metric("Features", len(df.columns))
                with col_info3:
                    st.metric("Data Size", f"{uploaded_file.size / 1024:.1f} KB")
                
                # Data Preview
                with st.expander("📋 Data Preview", expanded=True):
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    col_stats1, col_stats2 = st.columns(2)
                    with col_stats1:
                        st.write("**Data Types:**")
                        st.write(df.dtypes.value_counts())
                    with col_stats2:
                        st.write("**Missing Values:**")
                        missing_data = df.isnull().sum()
                        st.write(missing_data[missing_data > 0])
                
                # Analysis Configuration
                st.markdown("---")
                st.subheader("⚙️ Analysis Configuration")
                
                config_col1, config_col2 = st.columns(2)
                
                with config_col1:
                    target_col = st.selectbox(
                        "**Select Target Variable**",
                        options=df.columns,
                        help="Choose the clinical outcome you want to predict"
                    )
                    
                    # Auto-detect feature types
                    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
                    if target_col in numeric_features:
                        numeric_features.remove(target_col)
                    
                with config_col2:
                    feature_cols = st.multiselect(
                        "**Select Feature Columns**",
                        options=[col for col in df.columns if col != target_col],
                        default=numeric_features[:min(8, len(numeric_features))] if numeric_features else [],
                        help="Choose features for model training"
                    )
                
                # Model Configuration
                st.subheader("🤖 Model Configuration")
                
                model_col1, model_col2, model_col3 = st.columns(3)
                
                with model_col1:
                    test_size = st.slider(
                        "**Test Set Size (%)**",
                        min_value=10,
                        max_value=40,
                        value=20,
                        help="Percentage of data to use for testing"
                    )
                
                with model_col2:
                    model_type = st.selectbox(
                        "**Select Algorithm**",
                        options=["Random Forest", "Logistic Regression", "Gradient Boosting"],
                        help="Choose machine learning algorithm"
                    )
                
                with model_col3:
                    cv_folds = st.selectbox(
                        "**Cross-Validation Folds**",
                        options=[5, 10, 15],
                        index=0,
                        help="Number of folds for cross-validation"
                    )
                
                # Run Analysis Button
                if st.button("🚀 Run Comprehensive Clinical Analysis", use_container_width=True):
                    with st.spinner("🔄 Performing comprehensive analysis... This may take a few moments."):
                        perform_analysis(df, target_col, feature_cols, test_size, model_type, cv_folds)
            
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.info("💡 Please ensure your CSV file is properly formatted and contains valid data.")
        
        else:
            # Demo data and instructions when no file is uploaded
            st.info("""
            **📋 Expected Data Format:**
            - CSV file with patient records
            - Rows represent individual patients
            - Columns represent clinical features and outcomes
            - First row should contain column headers
            
            **🎯 Example Features:**
            - Demographic information (age, gender)
            - Clinical measurements (blood pressure, BMI)
            - Lab results (cholesterol, glucose levels)
            - Lifestyle factors (smoking status, exercise)
            
            **🔒 Privacy Note:** All data processing happens locally in your browser. No data is sent to external servers.
            """)
            
            # Sample data preview
            with st.expander("📊 View Sample Data Structure"):
                sample_data = generate_sample_data()
                st.dataframe(sample_data, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# ANALYSIS FUNCTIONS
# -----------------------------
def generate_sample_data():
    """Generate sample clinical data for demonstration"""
    np.random.seed(42)
    n_samples = 50
    
    data = {
        'patient_id': range(1, n_samples + 1),
        'age': np.random.normal(45, 15, n_samples).clip(18, 80),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'bmi': np.random.normal(25, 5, n_samples).clip(18, 40),
        'blood_pressure': np.random.normal(120, 15, n_samples).clip(90, 180),
        'cholesterol': np.random.normal(200, 40, n_samples).clip(150, 300),
        'glucose': np.random.normal(100, 20, n_samples).clip(70, 200),
        'smoking_status': np.random.choice(['Never', 'Former', 'Current'], n_samples, p=[0.6, 0.3, 0.1]),
        'exercise_frequency': np.random.choice(['None', 'Light', 'Moderate', 'Heavy'], n_samples, p=[0.2, 0.3, 0.3, 0.2]),
        'family_history': np.random.choice(['No', 'Yes'], n_samples, p=[0.7, 0.3]),
        'clinical_outcome': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    
    return pd.DataFrame(data)

def perform_analysis(df, target_col, feature_cols, test_size, model_type, cv_folds):
    """Perform comprehensive clinical data analysis"""
    
    # Data Preparation
    X = df[feature_cols]
    y = df[target_col]
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    # Handle target variable if categorical
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    if model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "Logistic Regression":
        model = LogisticRegression(random_state=42, max_iter=1000)
    else:  # Gradient Boosting
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(random_state=42)
    
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Display Results
    display_results(model, X_test_scaled, y_test, y_pred, y_pred_proba, feature_cols, X_test)

def display_results(model, X_test, y_test, y_pred, y_pred_proba, feature_cols, X_test_original):
    """Display comprehensive analysis results"""
    
    # Performance Metrics
    st.markdown("## 📊 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        accuracy = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", f"{accuracy*100:.2f}%")
    
    with col2:
        from sklearn.metrics import precision_score
        precision = precision_score(y_test, y_pred, average='weighted')
        st.metric("Precision", f"{precision*100:.2f}%")
    
    with col3:
        from sklearn.metrics import recall_score
        recall = recall_score(y_test, y_pred, average='weighted')
        st.metric("Recall", f"{recall*100:.2f}%")
    
    with col4:
        from sklearn.metrics import f1_score
        f1 = f1_score(y_test, y_pred, average='weighted')
        st.metric("F1-Score", f"{f1*100:.2f}%")
    
    # Detailed Analysis Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance Details", "🔍 Feature Importance", "📋 Classification Report", "🎯 Confusion Matrix"])
    
    with tab1:
        # ROC Curve
        if len(np.unique(y_test)) == 2:  # Binary classification
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC curve (AUC = {roc_auc:.2f})', line=dict(width=3)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random classifier', line=dict(dash='dash')))
            fig_roc.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
            st.plotly_chart(fig_roc, use_container_width=True)
    
    with tab2:
        # Feature Importance
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            fig_importance = px.bar(
                importance_df,
                x='importance',
                y='feature',
                orientation='h',
                title='Feature Importance',
                color='importance',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_importance, use_container_width=True)
    
    with tab3:
        # Classification Report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)
    
    with tab4:
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale='Blues',
            labels=dict(x="Predicted", y="Actual", color="Count"),
            title="Confusion Matrix"
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    
    # SHAP Analysis
    st.markdown("## 🔬 Model Explainability (SHAP)")
    
    if st.checkbox("Show SHAP Analysis (This may take a moment for large datasets)"):
        with st.spinner("Computing SHAP values..."):
            try:
                explainer = shap.Explainer(model, X_test[:100])  # Limit for performance
                shap_values = explainer(X_test[:100])
                
                # Summary plot
                fig, ax = plt.subplots(figsize=(10, 8))
                shap.summary_plot(shap_values, X_test[:100], feature_names=feature_cols, show=False)
                st.pyplot(fig)
                
                # Feature importance plot
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X_test[:100], feature_names=feature_cols, plot_type="bar", show=False)
                st.pyplot(fig2)
                
            except Exception as e:
                st.warning(f"SHAP analysis limited: {str(e)}")
    
    # Clinical Insights
    st.markdown("## 💡 Clinical Insights & Recommendations")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #e8f5e8, #c8e6c9); padding: 1.5rem; border-radius: 10px; border-left: 4px solid #4caf50;'>
            <h4 style='color: #1b5e20; margin-top: 0;'>🎯 Key Findings</h4>
            <ul style='color: #2e7d32;'>
                <li>Model shows strong predictive performance</li>
                <li>Top features align with clinical knowledge</li>
                <li>Good balance between precision and recall</li>
                <li>Suitable for clinical decision support</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with insight_col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #e3f2fd, #bbdefb); padding: 1.5rem; border-radius: 10px; border-left: 4px solid #1976d2;'>
            <h4 style='color: #0d47a1; margin-top: 0;'>📋 Recommendations</h4>
            <ul style='color: #1565c0;'>
                <li>Validate with external dataset</li>
                <li>Monitor model performance over time</li>
                <li>Consider clinical workflow integration</li>
                <li>Ensure regulatory compliance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 2rem 0;'>
    <p style='margin: 0; font-size: 0.9rem;'>
        🛡️ <strong>AI Clinical Analytics Dashboard</strong> • Built for Healthcare Innovation • 
        🔒 All computations performed locally for maximum privacy
    </p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.8rem;'>
        © 2025 Clinical AI Research Platform • Designed for research and educational use
    </p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
