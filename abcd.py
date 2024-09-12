import streamlit as st
import pandas as pd
import numpy as np
pip install plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.svm import SVC
import re  # Importing the regular expression library

# Streamlit App
st.set_page_config(page_title="ML Model Application", layout="wide")

st.title("Car Insurance Web Application")
st.write("Crew: Animesh Bhagya Shree Krishna Ritvik Varun")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", data.head())

    # Splitting the data
    train_df, test_df = train_test_split(data, test_size=0.2, stratify=data['is_claim'], random_state=42)

    # Function to extract numeric values from 'max_torque' and 'max_power'
    def extract_numeric_value(text):
        """Extracts the first numeric value from a string."""
        try:
            # Extract the first numeric value found in the string
            return float(re.findall(r'\d+', str(text))[0])
        except:
            return np.nan

    # Data Preprocessing
    train_df['policy_tenure'] = pd.to_numeric(train_df['policy_tenure'], errors='coerce')
    train_df['age_of_car'] = pd.to_numeric(train_df['age_of_car'], errors='coerce')
    train_df['age_of_policyholder'] = pd.to_numeric(train_df['age_of_policyholder'], errors='coerce')
    
    # Extract numeric values from 'max_torque' and 'max_power'
    train_df['max_torque'] = train_df['max_torque'].apply(extract_numeric_value)
    train_df['max_power'] = train_df['max_power'].apply(extract_numeric_value)
    train_df['displacement'] = pd.to_numeric(train_df['displacement'], errors='coerce')

    # Handle boolean columns
    boolean_columns = ['is_esc', 'is_adjustable_steering', 'is_tpms', 'is_parking_sensors', 'is_parking_camera', 
                       'is_front_fog_lights', 'is_rear_window_wiper', 'is_rear_window_washer', 
                       'is_rear_window_defogger', 'is_brake_assist', 'is_power_door_locks',
                       'is_central_locking', 'is_power_steering', 'is_driver_seat_height_adjustable', 
                       'is_day_night_rear_view_mirror', 'is_ecw', 'is_speed_alert']
    for col in boolean_columns:
        train_df[col] = train_df[col].apply(lambda x: 1 if x == 'Yes' else 0)

    # Outliers detection using box plots
    st.subheader("Outliers Detection")
    fig = make_subplots(rows=3, cols=2, subplot_titles=['policy_tenure', 'age_of_car', 'age_of_policyholder', 
                                                        'max_torque', 'max_power', 'displacement'])
    fig.add_trace(go.Box(y=train_df['policy_tenure'], name='policy_tenure', marker_color='blue'), row=1, col=1)
    fig.add_trace(go.Box(y=train_df['age_of_car'], name='age_of_car', marker_color='orange'), row=1, col=2)
    fig.add_trace(go.Box(y=train_df['age_of_policyholder'], name='age_of_policyholder', marker_color='green'), row=2, col=1)
    fig.add_trace(go.Box(y=train_df['max_torque'], name='max_torque', marker_color='red'), row=2, col=2)
    fig.add_trace(go.Box(y=train_df['max_power'], name='max_power', marker_color='purple'), row=3, col=1)
    fig.add_trace(go.Box(y=train_df['displacement'], name='displacement', marker_color='brown'), row=3, col=2)
    fig.update_layout(height=800, width=800, title_text="Outliers Detection", template='plotly_dark')
    st.plotly_chart(fig)

    # Remove outliers
    def remove_outliers(df, col):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
        return df

    for col in ['policy_tenure', 'age_of_car', 'age_of_policyholder', 'max_torque', 'max_power', 'displacement']:
        train_df = remove_outliers(train_df, col)

    # Define preprocessing pipeline
    numerical_cols = ['policy_tenure', 'age_of_car', 'age_of_policyholder', 'max_torque', 'max_power', 'displacement']
    categorical_cols = ['area_cluster', 'make', 'segment', 'model', 'fuel_type', 'engine_type', 
                        'rear_brakes_type', 'transmission_type', 'steering_type']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('scaler', StandardScaler()), ('power', PowerTransformer())]), numerical_cols),
            ('cat', OneHotEncoder(drop='first'), categorical_cols)
        ])

    # Prepare data for modeling
    X = train_df.drop(columns=['policy_id', 'is_claim'])
    y = train_df['is_claim']
    X_transformed = preprocessor.fit_transform(X)

    # Apply SMOTE for class imbalance
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X_transformed, y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)

    # Model Selection
    st.subheader("Select Model for Training")
    model_options = ["Random Forest", "Logistic Regression", "XGBoost", "Support Vector Classifier (SVC)"]
    model_choice = st.selectbox("Choose a model", model_options)

    # Train the selected model
    if model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_choice == "Logistic Regression":
        model = LogisticRegression()
    elif model_choice == "XGBoost":
        model = XGBClassifier()
    elif model_choice == "Support Vector Classifier (SVC)":
        model = SVC()

    try:
        model.fit(X_train, y_train)
        
        # Model Evaluation
        y_pred = model.predict(X_test)
        st.write("### Classification Report")
        st.text(classification_report(y_test, y_pred))

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        fig = px.imshow(conf_matrix, text_auto=True, title=f'Confusion Matrix for {model_choice}', template='plotly_dark')
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")
