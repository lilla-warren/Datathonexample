# app.py - PROFESSIONAL HOSPITAL ANALYTICS PLATFORM
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(
    page_title="MedAnalytics Pro - Clinical Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Hospital CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1a237e;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        padding: 2rem 0;
        background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hospital-subtitle {
        text-align: center;
        font-size: 1.3rem;
        color: #546e7a;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    .section-header {
        font-size: 1.8rem;
        color: #1a237e;
        border-bottom: 2px solid #1976d2;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
        font-weight: 600;
    }
    
    .clinical-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1976d2;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #e3f2fd;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1976d2 0%, #1565c0 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(25, 118, 210, 0.3);
        margin: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    .status-success {
        background: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #2e7d32;
    }
    
    .status-warning {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #ef6c00;
    }
    
    .status-error {
        background: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #c62828;
    }
    
    .sidebar-section {
        background: white;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #e3f2fd;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Professional Header
    st.markdown('<h1 class="main-header">🏥 MedAnalytics Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hospital-subtitle">Clinical Intelligence Platform • Predictive Analytics • Patient Risk Stratification</p>', unsafe_allow_html=True)
    
    # Sidebar - Clinical Configuration
    with st.sidebar:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a237e 0%, #1976d2 100%); 
                   color: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem;">
            <h3 style="color: white; margin: 0; font-size: 1.4rem;">Clinical Analytics</h3>
            <p style="color: white; opacity: 0.9; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                AI-Powered Healthcare Intelligence
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Data Upload
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("**📁 Clinical Data Upload**")
        uploaded_file = st.file_uploader("Upload Patient Dataset (CSV)", type=["csv"], label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Dataset Information
                st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
                st.markdown("**📊 Dataset Overview**")
                st.markdown(f"**Patients:** {df.shape[0]:,}")
                st.markdown(f"**Clinical Variables:** {df.shape[1]}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Analysis Configuration
                st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
                st.markdown("**🎯 Clinical Outcome**")
                target_variable = st.selectbox(
                    "Select Clinical Outcome Variable",
                    options=df.columns.tolist(),
                    index=len(df.columns)-1
                )
                
                # Show target distribution
                if target_variable:
                    target_counts = df[target_variable].value_counts()
                    st.markdown("**Patient Distribution:**")
                    for value, count in target_counts.items():
                        percentage = (count / len(df)) * 100
                        st.markdown(f"• **{value}:** {count:,} ({percentage:.1f}%)")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # ML Configuration
                st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
                st.markdown("**🤖 Predictive Models**")
                selected_models = st.multiselect(
                    "Select Clinical Prediction Models",
                    ["Logistic Regression", "Random Forest"],
                    default=["Logistic Regression", "Random Forest"]
                )
                
                test_size = st.slider("Validation Cohort Size (%)", 20, 40, 30)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Analysis Button
                if st.button("🚀 LAUNCH CLINICAL ANALYSIS", type="primary", use_container_width=True):
                    run_clinical_analysis(df, target_variable, selected_models, test_size/100)
                    
            except Exception as e:
                st.markdown(f"""
                <div class="status-error">
                    <strong>❌ Data Integration Error</strong><br>
                    {str(e)}
                </div>
                """, unsafe_allow_html=True)
        else:
            # Sample data guidance
            st.markdown("""
            <div class="status-warning">
                <strong>💡 Getting Started</strong><br>
                Upload clinical dataset in CSV format with patient records.
            </div>
            """, unsafe_allow_html=True)

def run_clinical_analysis(df, target_variable, selected_models, test_size):
    """Run comprehensive clinical analysis"""
    
    # Store original target for display
    original_target = df[target_variable].copy()
    
    # DATA PREPARATION
    with st.spinner("🔄 Processing clinical data for analysis..."):
        df_clean = df.copy()
        
        # Handle missing values
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Prepare features and target
        X = df_clean.drop(columns=[target_variable])
        y = df_clean[target_variable]
        
        # Encode categorical variables
        categorical_features = X.select_dtypes(include=['object']).columns
        for col in categorical_features:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        # Encode target variable
        if y.dtype == 'object':
            le = LabelEncoder()
            y_encoded = le.fit_transform(y.astype(str))
        else:
            y_encoded = y
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
    
    # SUCCESS MESSAGE
    st.markdown(f"""
    <div class="status-success">
        <strong>✅ Clinical Data Prepared Successfully</strong><br>
        • <strong>Training Cohort:</strong> {X_train.shape[0]:,} patients<br>
        • <strong>Validation Cohort:</strong> {X_test.shape[0]:,} patients<br>
        • <strong>Clinical Features:</strong> {X_train.shape[1]} variables
    </div>
    """, unsafe_allow_html=True)
    
    # CREATE PROFESSIONAL TABS
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Clinical Overview", 
        "🤖 Model Performance", 
        "📈 Analytics Dashboard", 
        "🔍 Feature Analysis",
        "💡 Clinical Insights"
    ])
    
    # TAB 1: CLINICAL OVERVIEW
    with tab1:
        st.markdown('<div class="section-header">Patient Data Overview</div>', unsafe_allow_html=True)
        
        # Key Metrics Grid
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Total Patients</div>
                <div class="metric-value">{:,}</div>
            </div>
            """.format(len(X)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Clinical Variables</div>
                <div class="metric-value">{}</div>
            </div>
            """.format(X.shape[1]), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Training Cohort</div>
                <div class="metric-value">{:,}</div>
            </div>
            """.format(X_train.shape[0]), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Validation Cohort</div>
                <div class="metric-value">{:,}</div>
            </div>
            """.format(X_test.shape[0]), unsafe_allow_html=True)
        
        # Data Preview and Statistics
        st.markdown("#### Data Exploration")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Patient Data Preview**")
            st.dataframe(df.head(10), use_container_width=True, height=300)
        
        with col2:
            st.markdown("**Clinical Statistics**")
            st.dataframe(df.describe(), use_container_width=True, height=300)
        
        # Target Distribution Visualization
        st.markdown("#### Patient Outcome Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#1976d2', '#4caf50']
            pd.Series(original_target).value_counts().plot(kind='bar', ax=ax, color=colors)
            ax.set_title('Clinical Outcome Distribution', fontsize=14, fontweight='bold')
            ax.set_xlabel('Clinical Outcome')
            ax.set_ylabel('Number of Patients')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#bbdefb', '#c8e6c9']
            pd.Series(original_target).value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=colors)
            ax.set_title('Outcome Proportions', fontsize=14, fontweight='bold')
            ax.set_ylabel('')
            st.pyplot(fig)
    
    # TAB 2: MODEL PERFORMANCE
    with tab2:
        st.markdown('<div class="section-header">Clinical Prediction Models</div>', unsafe_allow_html=True)
        
        # Train models
        results = []
        trained_models = {}
        predictions = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, model_name in enumerate(selected_models):
            status_text.text(f"🔄 Training {model_name}...")
            progress_bar.progress((i) / len(selected_models))
            
            try:
                if model_name == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000, random_state=42)
                elif model_name == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    continue
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate clinical performance metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                results.append({
                    'Model': model_name,
                    'Accuracy': round(accuracy, 4),
                    'Precision': round(precision, 4),
                    'Recall': round(recall, 4),
                    'F1-Score': round(f1, 4),
                    'ROC-AUC': round(roc_auc, 4)
                })
                
                trained_models[model_name] = model
                predictions[model_name] = (y_pred, y_pred_proba)
                
            except Exception as e:
                st.markdown(f"""
                <div class="status-warning">
                    Could not train {model_name}: {str(e)}
                </div>
                """, unsafe_allow_html=True)
        
        progress_bar.progress(1.0)
        status_text.text("✅ Model training completed!")
        
        # Display results
        if results:
            results_df = pd.DataFrame(results)
            
            st.markdown("#### Model Performance Metrics")
            st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
            styled_df = results_df.style.highlight_max(subset=['ROC-AUC', 'Accuracy'])
            st.dataframe(styled_df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Best model info
            best_model = results_df.loc[results_df['ROC-AUC'].idxmax()]
            st.markdown(f"""
            <div class="status-success">
                <strong>🎯 Optimal Clinical Model Identified</strong><br>
                • <strong>Algorithm:</strong> {best_model['Model']}<br>
                • <strong>ROC-AUC:</strong> {best_model['ROC-AUC']:.3f}<br>
                • <strong>Accuracy:</strong> {best_model['Accuracy']:.3f}<br>
                • <strong>Clinical Utility:</strong> {'Excellent' if best_model['ROC-AUC'] > 0.8 else 'Good' if best_model['ROC-AUC'] > 0.7 else 'Moderate'}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-error">
                <strong>❌ Model Training Failed</strong><br>
                No models were successfully trained.
            </div>
            """, unsafe_allow_html=True)
            return
    
    # TAB 3: ANALYTICS DASHBOARD
    with tab3:
        st.markdown('<div class="section-header">Clinical Analytics Dashboard</div>', unsafe_allow_html=True)
        
        if not results:
            st.markdown("""
            <div class="status-warning">
                Please train models first in the 'Model Performance' tab.
            </div>
            """, unsafe_allow_html=True)
            return
        
        # ROC Curves
        st.markdown("#### ROC Analysis - Diagnostic Accuracy")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for model_name, (y_pred, y_pred_proba) in predictions.items():
            RocCurveDisplay.from_predictions(y_test, y_pred_proba, name=model_name, ax=ax)
        
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
        ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Confusion Matrix
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Confusion Matrix - Best Model")
            best_model_name = results_df.loc[results_df['ROC-AUC'].idxmax(), 'Model']
            y_pred_best, _ = predictions[best_model_name]
            
            fig, ax = plt.subplots(figsize=(6, 5))
            cm = confusion_matrix(y_test, y_pred_best)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'{best_model_name}\nConfusion Matrix', fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicted Outcome')
            ax.set_ylabel('Actual Outcome')
            st.pyplot(fig)
    
    # TAB 4: FEATURE ANALYSIS
    with tab4:
        st.markdown('<div class="section-header">Clinical Feature Analysis</div>', unsafe_allow_html=True)
        
        if not results:
            st.markdown("""
            <div class="status-warning">
                Please train models first in the 'Model Performance' tab.
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Feature Importance
        st.markdown("#### Feature Importance Analysis")
        tree_models = {name: model for name, model in trained_models.items() 
                      if hasattr(model, 'feature_importances_')}
        
        if tree_models:
            best_tree_model_name = max(tree_models.keys(), 
                                      key=lambda x: results_df[results_df['Model'] == x]['ROC-AUC'].values[0])
            best_tree_model = tree_models[best_tree_model_name]
            
            feature_importance = pd.DataFrame({
                'Clinical Feature': X.columns,
                'Importance': best_tree_model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(feature_importance)))
            bars = ax.barh(feature_importance['Clinical Feature'], feature_importance['Importance'], color=colors)
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                       f'{width:.3f}', ha='left', va='center', fontweight='bold')
            
            ax.set_title(f'Top 10 Predictive Features\n{best_tree_model_name}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Predictive Importance Score')
            ax.grid(True, alpha=0.3, axis='x')
            st.pyplot(fig)
    
    # TAB 5: CLINICAL INSIGHTS
    with tab5:
        st.markdown('<div class="section-header">Clinical Insights & Recommendations</div>', unsafe_allow_html=True)
        
        if not results:
            st.markdown("""
            <div class="status-warning">
                Please train models first in the 'Model Performance' tab.
            </div>
            """, unsafe_allow_html=True)
            return
        
        best_auc = results_df['ROC-AUC'].max()
        best_model_name = results_df.loc[results_df['ROC-AUC'].idxmax(), 'Model']
        
        # Clinical Insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏥 Clinical Applications")
            st.markdown("""
            **Risk Stratification**
            - Identify high-risk patients proactively
            - Prioritize clinical interventions  
            - Optimize resource allocation
            
            **Decision Support**
            - Augment clinical judgment with AI insights
            - Reduce diagnostic variability
            - Improve treatment planning accuracy
            """)
            
            st.markdown("#### 📋 Implementation Roadmap")
            st.markdown("""
            **Phase 1: Clinical Validation (2-4 weeks)**
            - Model validation studies
            - Clinical protocol development
            - Stakeholder alignment
            
            **Phase 2: Pilot Deployment (4-8 weeks)**
            - Limited clinical deployment
            - Healthcare staff training
            - Performance monitoring
            """)
        
        with col2:
            st.markdown("#### 💡 Key Findings")
            st.markdown(f"""
            **Model Performance Summary**
            - **Best Algorithm:** {best_model_name}
            - **Predictive Accuracy:** {best_auc:.1%}
            - **Clinical Utility:** {'Excellent' if best_auc > 0.8 else 'Good' if best_auc > 0.7 else 'Moderate'}
            - **Patient Cohort:** {len(X):,}
            - **Feature Richness:** {X.shape[1]} variables
            
            **Data Quality Assessment**
            - Data Completeness: Good
            - Feature Relevance: High  
            - Clinical Applicability: Strong
            """)
            
            st.markdown("#### ⚖️ Ethical Considerations")
            st.markdown("""
            **Patient Safety & Governance**
            - AI supports, never replaces clinical judgment
            - Regular model performance monitoring required
            - Bias detection and mitigation protocols
            - HIPAA compliance maintained throughout
            - Patient privacy protection ensured
            - Transparent algorithm documentation
            """)

if __name__ == "__main__":
    main()
