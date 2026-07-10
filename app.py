import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'model_results' not in st.session_state:
    st.session_state.model_results = None

# Sidebar navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Select a Page",
    [
        "Dashboard",
        "Data Overview",
        "Exploratory Data Analysis",
        "Model Training",
        "Hyperparameter Tuning",
        "Model Comparison",
        "Explainable AI",
        "Predict New Customer",
        "Reports"
    ]
)

# Load data function
@st.cache_data
def load_data():
    from config import DATA_PATH
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        return df
    return None

# Preprocess data function
@st.cache_data
def preprocess_data(df):
    from src.preproessing import DataPreprocessor
    processor = DataPreprocessor("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df_processed = processor.preprocess()
    return df_processed

# Feature engineering function
@st.cache_data
def feature_engineering(df):
    from src.feature_engineering import FeatureEngineering
    feature = FeatureEngineering(df)
    X_train, X_test, y_train, y_test = feature.run()
    return X_train, X_test, y_train, y_test

# Dashboard Page
def dashboard_page():
    st.markdown('<h1 class="main-title">Customer Churn Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Customers", "7,043")
    with col2:
        st.metric("Churn Rate", "26.5%")
    with col3:
        st.metric("Models Available", "4")
    
    st.markdown("---")
    
    st.subheader("🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Load & Preprocess Data", key="load_data"):
            df = load_data()
            if df is not None:
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.success("Data loaded successfully!")
            else:
                st.error("Dataset not found!")
    
    with col2:
        if st.button("Train All Models", key="train_models", disabled=not st.session_state.data_loaded):
            st.info("Navigate to Model Training page to train models")
    
    with col3:
        if st.button("View Best Model", key="view_best"):
            if os.path.exists("models/best_model.pkl"):
                st.success("Best model is available!")
            else:
                st.warning("Train models first")
    
    st.markdown("---")
    
    st.subheader("📈 Project Overview")
    
    overview_data = {
        "Feature": ["Dataset Size", "Features", "Target Variable", "Models", "Evaluation Metrics"],
        "Value": ["7,043 rows", "20 features", "Churn (Binary)", "4 ML Models", "6 Metrics"]
    }
    
    df_overview = pd.DataFrame(overview_data)
    st.table(df_overview)

# Data Overview Page
def data_overview_page():
    st.markdown('<h1 class="main-title">Data Overview</h1>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first from the Dashboard page")
        if st.button("Load Data Now"):
            df = load_data()
            if df is not None:
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.success("Data loaded successfully!")
                st.rerun()
        return
    
    df = st.session_state.df
    
    tab1, tab2, tab3, tab4 = st.tabs(["Dataset", "Statistics", "Missing Values", "Class Distribution"])
    
    with tab1:
        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head(10))
        st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
        st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
    
    with tab2:
        st.subheader("📊 Statistical Summary")
        st.dataframe(df.describe(include='all'))
        
        st.subheader("Data Types")
        dtypes_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.values,
            "Non-Null Count": df.count().values
        })
        st.dataframe(dtypes_df)
    
    with tab3:
        st.subheader("🔍 Missing Values")
        missing = df.isnull().sum()
        missing_df = pd.DataFrame({
            "Column": missing.index,
            "Missing Count": missing.values,
            "Missing Percentage": (missing.values / len(df) * 100).round(2)
        })
        st.dataframe(missing_df)
        
        if missing.sum() == 0:
            st.success("No missing values found!")
        else:
            st.warning(f"Found {missing.sum()} missing values")
    
    with tab4:
        st.subheader("🎯 Class Distribution")
        if 'Churn' in df.columns:
            churn_counts = df['Churn'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_pie = px.pie(
                    values=churn_counts.values,
                    names=churn_counts.index,
                    title="Churn Distribution",
                    color_discrete_sequence=['#1f77b4', '#ff7f0e']
                )
                st.plotly_chart(fig_pie, width='stretch')
            
            with col2:
                fig_bar = px.bar(
                    x=churn_counts.index,
                    y=churn_counts.values,
                    title="Churn Count",
                    color=churn_counts.index,
                    color_discrete_sequence=['#1f77b4', '#ff7f0e']
                )
                st.plotly_chart(fig_bar, width='stretch')
            
            st.write(f"**Churn Rate:** {(churn_counts[1] / len(df) * 100):.2f}%")
        else:
            st.warning("Churn column not found in dataset")

# EDA Page
def eda_page():
    st.markdown('<h1 class="main-title">Exploratory Data Analysis</h1>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first from the Dashboard page")
        return
    
    df = st.session_state.df
    
    tab1, tab2, tab3, tab4 = st.tabs(["Histograms", "Correlation Heatmap", "Boxplots", "Pie Charts"])
    
    with tab1:
        st.subheader("📊 Numerical Feature Distributions")
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numerical_cols:
            for col in numerical_cols[:6]:  # Show first 6 numerical columns
                fig = px.histogram(df, x=col, title=f"Distribution of {col}", nbins=30)
                st.plotly_chart(fig, width='stretch')
        else:
            st.info("No numerical columns found")
    
    with tab2:
        st.subheader("🔥 Correlation Heatmap")
        numerical_df = df.select_dtypes(include=[np.number])
        
        if len(numerical_df.columns) > 1:
            corr_matrix = numerical_df.corr()
            
            fig = px.imshow(
                corr_matrix,
                title="Correlation Heatmap",
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("Not enough numerical columns for correlation")
    
    with tab3:
        st.subheader("📦 Boxplots - Outlier Detection")
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numerical_cols:
            selected_col = st.selectbox("Select Feature", numerical_cols)
            fig = px.box(df, y=selected_col, title=f"Boxplot of {selected_col}")
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No numerical columns found")
    
    with tab4:
        st.subheader("🥧 Categorical Feature Distributions")
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_cols:
            selected_col = st.selectbox("Select Categorical Feature", categorical_cols)
            
            value_counts = df[selected_col].value_counts()
            
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Distribution of {selected_col}"
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No categorical columns found")

# Model Training Page
def model_training_page():
    st.markdown('<h1 class="main-title">Model Training</h1>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load and preprocess data first")
        return
    
    st.subheader("🤖 Available Models")
    
    models = {
        "Logistic Regression": "A linear model for binary classification",
        "Decision Tree": "A tree-based model that splits data based on features",
        "Random Forest": "An ensemble of decision trees",
        "XGBoost": "Gradient boosting algorithm for high performance"
    }
    
    for model_name, description in models.items():
        with st.expander(f"**{model_name}**"):
            st.write(description)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Training Configuration")
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random State", 0, 100, 42)
        apply_smote = st.checkbox("Apply SMOTE (Class Balancing)", value=True)
    
    with col2:
        st.subheader("📊 Training Status")
        if st.button("Train All Models", key="train_all"):
            with st.spinner("Training models... This may take a few minutes"):
                try:
                    # Preprocess data
                    from src.preproessing import DataPreprocessor
                    processor = DataPreprocessor("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
                    df_processed = processor.preprocess()
                    
                    # Feature engineering
                    from src.feature_engineering import FeatureEngineering
                    feature = FeatureEngineering(df_processed)
                    X_train, X_test, y_train, y_test = feature.run()
                    
                    # Store in session state
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    
                    # Train models
                    from src.model_training import ModelTrainer
                    trainer = ModelTrainer(X_train, X_test, y_train, y_test)
                    trainer.run()
                    
                    # Load results
                    if os.path.exists("outputs/metrics/model_metrics.csv"):
                        st.session_state.model_results = pd.read_csv("outputs/metrics/model_metrics.csv")
                        st.session_state.models_trained = True
                    
                    st.success("All models trained successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
    
    if st.session_state.models_trained and st.session_state.model_results is not None:
        st.markdown("---")
        st.subheader("✅ Training Results")
        st.dataframe(st.session_state.model_results)

# Hyperparameter Tuning Page
def hyperparameter_tuning_page():
    st.markdown('<h1 class="main-title">Hyperparameter Tuning</h1>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.warning("Please train models first from the Model Training page")
        return
    
    st.subheader("🔧 GridSearchCV Configuration")
    
    model_to_tune = st.selectbox(
        "Select Model to Tune",
        ["Random Forest", "XGBoost"]
    )
    
    if model_to_tune == "Random Forest":
        st.write("**Random Forest Parameters:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            n_estimators = st.multiselect("N Estimators", [50, 100, 200], [100, 200])
        with col2:
            max_depth = st.multiselect("Max Depth", [5, 10, 15], [5, 10, 15])
        with col3:
            min_samples_split = st.multiselect("Min Samples Split", [2, 5], [2, 5])
        
        if st.button("Tune Random Forest"):
            st.info("Hyperparameter tuning is performed during model training. Check Model Training results.")
    
    elif model_to_tune == "XGBoost":
        st.write("**XGBoost Parameters:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            n_estimators = st.multiselect("N Estimators", [100, 200], [100, 200])
        with col2:
            max_depth = st.multiselect("Max Depth", [4, 6], [4, 6])
        with col3:
            learning_rate = st.multiselect("Learning Rate", [0.05, 0.1], [0.05, 0.1])
        
        if st.button("Tune XGBoost"):
            st.info("Hyperparameter tuning is performed during model training. Check Model Training results.")
    
    st.markdown("---")
    
    st.subheader("📊 Best Parameters Found")
    
    if st.session_state.model_results is not None:
        tuned_models = st.session_state.model_results[st.session_state.model_results['Model'].str.contains('Tuned')]
        
        if not tuned_models.empty:
            st.dataframe(tuned_models)
        else:
            st.info("No tuned models found yet")

# Model Comparison Page
def model_comparison_page():
    st.markdown('<h1 class="main-title">Model Comparison</h1>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained or st.session_state.model_results is None:
        st.warning("Please train models first from the Model Training page")
        return
    
    results = st.session_state.model_results
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "Cross Validation"])
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC', 'CrossValidation']
    tabs = [tab1, tab2, tab3, tab4, tab5, tab6]
    
    for metric, tab in zip(metrics, tabs):
        with tab:
            st.subheader(f"📊 {metric} Comparison")
            
            fig = px.bar(
                results,
                x='Model',
                y=metric,
                title=f"{metric} by Model",
                color=metric,
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, width='stretch')
            
            # Show table
            st.subheader(f"📋 {metric} Rankings")
            ranked = results.sort_values(by=metric, ascending=False)
            st.dataframe(ranked[['Model', metric]])
    
    st.markdown("---")
    
    st.subheader("🏆 Overall Best Model")
    best_model = results.loc[results['Accuracy'].idxmax()]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Best Model", best_model['Model'])
        st.metric("Accuracy", f"{best_model['Accuracy']:.4f}")
    
    with col2:
        st.metric("ROC AUC", f"{best_model['ROC']:.4f}")
        st.metric("F1 Score", f"{best_model['F1']:.4f}")

# Explainable AI Page
def explainable_ai_page():
    st.markdown('<h1 class="main-title">Explainable AI (SHAP)</h1>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.warning("Please train models first from the Model Training page")
        return
    
    if not os.path.exists("models/best_model.pkl"):
        st.warning("Best model not found. Please train models first.")
        return
    
    st.subheader("🔍 SHAP (SHapley Additive exPlanations)")
    
    st.info("SHAP values explain how each feature contributes to the model's predictions.")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Summary Plot", "Waterfall Plot", "Feature Importance", "Dependence Plot"])
    
    with tab1:
        st.subheader("📊 SHAP Summary Plot")
        if os.path.exists("outputs/shap/shap_summary.png"):
            st.image("outputs/shap/shap_summary.png")
        else:
            st.info("Generate SHAP plots by running the explainability module")
            if st.button("Generate SHAP Plots"):
                try:
                    import joblib
                    from src.explainability import Explainability
                    
                    best_model = joblib.load("models/best_model.pkl")
                    explainer = Explainability(best_model, st.session_state.X_train, st.session_state.X_test)
                    explainer.run()
                    st.success("SHAP plots generated!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating SHAP plots: {str(e)}")
    
    with tab2:
        st.subheader("💧 SHAP Waterfall Plot")
        if os.path.exists("outputs/shap/shap_waterfall.png"):
            st.image("outputs/shap/shap_waterfall.png")
        else:
            st.info("Generate SHAP plots first")
    
    with tab3:
        st.subheader("📊 Feature Importance")
        if os.path.exists("outputs/shap/shap_feature_importance.png"):
            st.image("outputs/shap/shap_feature_importance.png")
        else:
            st.info("Generate SHAP plots first")
    
    with tab4:
        st.subheader("🔗 SHAP Dependence Plot")
        if os.path.exists("outputs/shap/shap_dependence.png"):
            st.image("outputs/shap/shap_dependence.png")
        else:
            st.info("Generate SHAP plots first")

# Predict New Customer Page
def predict_new_customer_page():
    st.markdown('<h1 class="main-title">Predict New Customer</h1>', unsafe_allow_html=True)
    
    if not os.path.exists("models/best_model.pkl"):
        st.warning("Please train models first from the Model Training page")
        return
    
    prediction_method = st.radio(
        "Choose Prediction Method",
        ["Upload CSV", "Manual Entry"],
        horizontal=True
    )
    
    if prediction_method == "Upload CSV":
        st.subheader("📁 Upload CSV File")
        
        uploaded_file = st.file_uploader("Upload a CSV file with customer data", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df_new = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
                st.dataframe(df_new.head())
                
                if st.button("Make Predictions"):
                    with st.spinner("Making predictions..."):
                        try:
                            from src.predict import PredictionPreprocessor
                            
                            # Initialize preprocessor
                            preprocessor = PredictionPreprocessor()
                            
                            # Make predictions
                            results = preprocessor.predict(df_new)
                            
                            # Display results
                            st.success("Predictions completed!")
                            
                            # Combine original data with predictions
                            df_results = df_new.copy()
                            df_results['Prediction'] = results['Prediction'].map({0: 'No', 1: 'Yes'})
                            df_results['Probability'] = results['Probability'].round(4)
                            df_results['Churn_Risk'] = results['Churn_Risk']
                            
                            st.subheader("📊 Prediction Results")
                            st.dataframe(df_results)
                            
                            # Summary statistics
                            st.subheader("📈 Summary")
                            churn_count = (results['Prediction'] == 1).sum()
                            total_count = len(results)
                            churn_rate = (churn_count / total_count * 100) if total_count > 0 else 0
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Customers", total_count)
                            with col2:
                                st.metric("Predicted Churn", churn_count)
                            with col3:
                                st.metric("Churn Rate", f"{churn_rate:.2f}%")
                            
                            # Download results
                            csv = df_results.to_csv(index=False)
                            st.download_button(
                                label="Download Predictions",
                                data=csv,
                                file_name="customer_predictions.csv",
                                mime="text/csv"
                            )
                            
                        except Exception as e:
                            st.error(f"Error making predictions: {str(e)}")
                            import traceback
                            st.error(traceback.format_exc())
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    else:
        st.subheader("✏️ Manual Entry")
        
        st.info("Enter customer details for prediction")
        
        # Full form matching the dataset structure
        col1, col2, col3 = st.columns(3)
        
        with col1:
            customer_id = st.text_input("Customer ID", "CUST-NEW")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.number_input("Tenure (months)", 0, 72, 12)
        
        with col2:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        
        with col3:
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", 
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
            total_charges = st.number_input("Total Charges", 0.0, 10000.0, tenure * monthly_charges)
        
        if st.button("Predict Churn"):
            try:
                from src.predict import predict_from_dict
                
                # Create customer data dictionary
                customer_data = {
                    "customerID": customer_id,
                    "gender": gender,
                    "SeniorCitizen": senior_citizen,
                    "Partner": partner,
                    "Dependents": dependents,
                    "tenure": tenure,
                    "PhoneService": phone_service,
                    "MultipleLines": multiple_lines,
                    "InternetService": internet_service,
                    "OnlineSecurity": online_security,
                    "OnlineBackup": online_backup,
                    "DeviceProtection": device_protection,
                    "TechSupport": tech_support,
                    "StreamingTV": streaming_tv,
                    "StreamingMovies": streaming_movies,
                    "Contract": contract,
                    "PaperlessBilling": paperless_billing,
                    "PaymentMethod": payment_method,
                    "MonthlyCharges": monthly_charges,
                    "TotalCharges": total_charges
                }
                
                # Make prediction
                results = predict_from_dict(customer_data)
                
                # Display results
                prediction = "Yes" if results['Prediction'][0] == 1 else "No"
                probability = results['Probability'][0]
                risk_level = results['Churn_Risk'][0]
                
                st.success("Prediction completed!")
                
                # Display prediction card
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Churn Prediction", prediction)
                with col2:
                    st.metric("Probability", f"{probability:.4f}")
                with col3:
                    st.metric("Risk Level", risk_level)
                
                # Risk level color coding
                if risk_level == "High":
                    st.error("⚠️ High churn risk - Immediate attention recommended")
                elif risk_level == "Medium":
                    st.warning("⚡ Medium churn risk - Monitor customer")
                else:
                    st.success("✅ Low churn risk - Customer stable")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

# Reports Page
def reports_page():
    st.markdown('<h1 class="main-title">Reports & Downloads</h1>', unsafe_allow_html=True)
    
    st.subheader("📥 Download Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📊 Metrics")
        if os.path.exists("outputs/metrics/model_metrics.csv"):
            with open("outputs/metrics/model_metrics.csv", 'rb') as f:
                st.download_button(
                    label="Download Model Metrics",
                    data=f,
                    file_name="model_metrics.csv",
                    mime="text/csv"
                )
        else:
            st.info("No metrics available")
        
        if os.path.exists("outputs/eda/dataset_summary.csv"):
            with open("outputs/eda/dataset_summary.csv", 'rb') as f:
                st.download_button(
                    label="Download Dataset Summary",
                    data=f,
                    file_name="dataset_summary.csv",
                    mime="text/csv"
                )
        else:
            st.info("No dataset summary available")
    
    with col2:
        st.subheader("🔮 Predictions")
        prediction_files = []
        if os.path.exists("outputs/predictions"):
            prediction_files = [f for f in os.listdir("outputs/predictions") if f.endswith('.csv')]
        
        if prediction_files:
            for pred_file in prediction_files:
                with open(f"outputs/predictions/{pred_file}", 'rb') as f:
                    st.download_button(
                        label=f"Download {pred_file}",
                        data=f,
                        file_name=pred_file,
                        mime="text/csv"
                    )
        else:
            st.info("No predictions available")
    
    with col3:
        st.subheader("📈 Graphs")
        graph_files = []
        if os.path.exists("outputs/plots"):
            graph_files = [f for f in os.listdir("outputs/plots") if f.endswith('.png')]
        
        if graph_files:
            for graph_file in graph_files:
                with open(f"outputs/plots/{graph_file}", 'rb') as f:
                    st.download_button(
                        label=f"Download {graph_file}",
                        data=f,
                        file_name=graph_file,
                        mime="image/png"
                    )
        else:
            st.info("No graphs available")
    
    st.markdown("---")
    
    st.subheader("📋 Available Reports Summary")
    
    report_data = {
        "Report Type": ["Model Metrics", "Dataset Summary", "Predictions", "Graphs", "SHAP Plots"],
        "Status": [
            "✅ Available" if os.path.exists("outputs/metrics/model_metrics.csv") else "❌ Not Available",
            "✅ Available" if os.path.exists("outputs/eda/dataset_summary.csv") else "❌ Not Available",
            f"✅ {len(prediction_files)} files" if prediction_files else "❌ Not Available",
            f"✅ {len(graph_files)} files" if graph_files else "❌ Not Available",
            "✅ Available" if os.path.exists("outputs/shap") else "❌ Not Available"
        ]
    }
    
    df_reports = pd.DataFrame(report_data)
    st.table(df_reports)

# Main app logic
if page == "Dashboard":
    dashboard_page()
elif page == "Data Overview":
    data_overview_page()
elif page == "Exploratory Data Analysis":
    eda_page()
elif page == "Model Training":
    model_training_page()
elif page == "Hyperparameter Tuning":
    hyperparameter_tuning_page()
elif page == "Model Comparison":
    model_comparison_page()
elif page == "Explainable AI":
    explainable_ai_page()
elif page == "Predict New Customer":
    predict_new_customer_page()
elif page == "Reports":
    reports_page()

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Customer Churn Prediction Dashboard © 2024</p>", unsafe_allow_html=True)
