import streamlit as st
import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import google.generativeai as genai
import base64
import os
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------------------------------------------
# Firebase Firestore Initialization (avoiding multiple initializations)
# ------------------------------------------------------------------
if not firebase_admin._apps:
    # Replace "serviceAccountKey.json" with the path to your service account JSON
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

# ------------------------------------------------------------------
# Streamlit Page Configuration
# ------------------------------------------------------------------
st.set_page_config(page_title="AQI Data Analyzer", layout="wide")

# ------------------------------------------------------------------
# API and Gemini Configuration
# ------------------------------------------------------------------
BASE_URL = "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
API_KEY = "579b464db66ec23bdd000001586fe5bfeda64e5c55d27328dcc242a8"
GEMINI_MODEL_NAME = "gemini-1.5-pro"

def configure_gemini():
    """Configure Gemini API with the provided API key."""
    try:
        # Replace with your actual Gemini API key
        genai.configure(api_key="AIzaSyCGkOX2-g8Iw7Q3R5u5bOZ3DHAmV-52tus")
    except Exception as e:
        st.error("Gemini API key not configured. AI features disabled.")

# ------------------------------------------------------------------
# Data Fetching and Parsing Functions
# ------------------------------------------------------------------
def fetch_aqi_data(country="India", state="Delhi", city="Delhi", limit=10000):
    """
    Fetches AQI data from the API using provided filter parameters.
    """
    params = {
        'api-key': API_KEY,
        'format': 'json',
        'limit': limit,
        'filters[country]': country,
        'filters[state]': state,
        'filters[city]': city
    }
    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    return response.json()

def parse_aqi_data(raw_data):
    """Parses the JSON data into a pandas DataFrame and cleans it."""
    if 'records' not in raw_data or not raw_data['records']:
        raise ValueError("No records found in the API response. Please check your filter parameters.")

    df = pd.DataFrame(raw_data['records'])

    # Convert numerical fields relevant to AQI analysis
    numeric_cols = ['min_value', 'max_value', 'avg_value']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing target values and fill missing numeric fields
    df.dropna(subset=['avg_value'], inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.drop_duplicates(inplace=True)
    
    return df

# ------------------------------------------------------------------
# Firestore Interaction Functions
# ------------------------------------------------------------------
def save_aqi_data(city, state, data):
    """
    Saves a single AQI record in Firestore.
    'data' is a dictionary representing one record of AQI data.
    """
    aqi_collection = db.collection('aqi_data')
    record = {
        'city': city,
        'state': state,
        **data,
        'timestamp': firestore.SERVER_TIMESTAMP  # auto-add the insertion time
    }
    aqi_collection.add(record)

def get_aqi_data_by_city(city):
    """
    Retrieves all AQI records for the specified city from Firestore.
    Returns a list of dictionaries.
    """
    aqi_collection = db.collection('aqi_data')
    query = aqi_collection.where('city', '==', city)
    docs = query.get()
    return [doc.to_dict() for doc in docs]

# ------------------------------------------------------------------
# Analysis and Visualization Functions
# ------------------------------------------------------------------
def generate_ai_insights(df, city, state):
    """Generates AI-powered insights using Gemini."""
    st.subheader("🤖 AI-Powered Air Quality Recommendations")
    
    # Data summary for generating insights
    pollutant_stats = df.groupby('pollutant_id')['avg_value'].agg(['mean', 'max'])
    dominant_pollutant = pollutant_stats['mean'].idxmax() if not pollutant_stats.empty else "Unknown"
    avg_aqi = df['avg_value'].mean()
    max_aqi = df['avg_value'].max()
    
    # Expanded prompt for more comprehensive recommendations
    prompt = f"""
Act as a seasoned environmental scientist specializing in air quality analysis. 

You have been provided with the following data for {city}, {state}:

Data Summary:
- Dominant Pollutant: {dominant_pollutant}
- Average AQI: {avg_aqi:.1f}
- Maximum AQI: {max_aqi:.1f}
- Top 3 Pollutants: {', '.join(pollutant_stats.nlargest(3, 'mean').index.tolist()) if not pollutant_stats.empty else 'N/A'}

In addition, consider the broader context: historical trends in AQI for {city} over the past several years, local government and community initiatives, public awareness campaigns, and planned or ongoing policy measures aimed at reducing air pollution. Provide a detailed, holistic analysis of the air quality situation and deliver comprehensive recommendations for short-term and long-term improvement.

Organize your output into the following sections and keep it short:

1. Historical Trends
   - Summarize how air pollution levels have changed over the past few years in {city}.
   - Mention any significant turning points or major events that impacted air quality.

2. Government and Community Initiatives
   - Highlight ongoing or planned local government actions, regulations, or infrastructure projects.
   - Discuss community-based programs or NGO efforts.

3. Immediate Actions
   - List 3 specific, actionable bullet points for short-term measures to immediately reduce the impact of {dominant_pollutant}.

4. Long-term Solutions
   - Outline strategic and sustainable approaches for improving air quality over time.

5. Health Advisories
   - Offer clear guidance to protect public health, especially for vulnerable populations.

6. Awareness and Education
   - Suggest ways to increase public engagement and knowledge about air pollution, including historical context.

7. Policy Recommendations
   - Propose targeted policy changes or regulatory interventions that local authorities can implement.

8. Community Initiatives
   - Suggest community-based projects or public awareness campaigns to promote cleaner air. 

Additional Guidelines:
- Use a balanced approach that includes both policy-level and community-level solutions.
- Employ data-driven insights and reference established environmental best practices.
- Write in accessible language while still including relevant technical details.
"""

    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        response = model.generate_content(prompt)
        with st.expander("View Detailed Recommendations"):
            st.markdown(response.text)
        st.success("AI-generated recommendations ready!")
    except Exception as e:
        st.error(f"Failed to generate insights: {str(e)}")
        st.info("Here are some general recommendations:")
        st.markdown("""
        - Avoid outdoor activities during peak pollution hours
        - Use public transportation whenever possible
        - Support green infrastructure initiatives
        """)

def perform_eda(df):
    """Performs exploratory data analysis with visualizations."""
    st.subheader("Data Overview")
    st.dataframe(df.head())
    
    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe())
    
    st.subheader("Data Visualizations")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Distribution of Pollutant Average")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(df['avg_value'], kde=True, bins=20, ax=ax)
        ax.set_title("Distribution of Pollutant Average")
        ax.set_xlabel("Pollutant Average Value")
        st.pyplot(fig)
    
    with col2:
        st.write("Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        corr = df[['min_value', 'max_value', 'avg_value']].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title("Feature Correlation Heatmap")
        st.pyplot(fig)
    
    st.write("Min vs. Max Pollutant Values")
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'pollutant_id' in df.columns:
        sns.scatterplot(data=df, x='min_value', y='max_value', hue='pollutant_id', ax=ax)
    else:
        sns.scatterplot(data=df, x='min_value', y='max_value', ax=ax)
    ax.set_title("Min vs. Max Pollutant Values")
    ax.set_xlabel("Min Pollutant Value")
    ax.set_ylabel("Max Pollutant Value")
    st.pyplot(fig)

def train_model(df):
    """
    Trains a Random Forest Regressor with hyperparameter tuning
    to predict AQI (avg_value) based on min_value and max_value.
    """
    st.subheader("Model Training Results")
    
    target = 'avg_value'
    X = df[['min_value', 'max_value']]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)

    with st.spinner("Tuning hyperparameters and training model..."):
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(estimator=rf,
                                   param_grid=param_grid,
                                   cv=3,
                                   scoring='neg_mean_squared_error',
                                   n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Train MSE", f"{train_mse:.2f}")
            st.metric("Test MSE", f"{test_mse:.2f}")
        with col2:
            st.metric("Train R²", f"{train_r2:.2f}")
            st.metric("Test R²", f"{test_r2:.2f}")
        
        # Removed the line that shows best hyperparameters to hide from UI
        # st.write("Best hyperparameters found:", grid_search.best_params_)
    
    return best_model, X_test, y_test

def visualize_results(model, X_test, y_test):
    """Generates residual plots and actual vs. predicted plots."""
    st.subheader("Model Visualization")
    y_pred = model.predict(X_test)
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Residual Plot")
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=y_pred, y=residuals, ax=ax)
        ax.axhline(y=0, color='red', linestyle='--')
        ax.set_title("Residual Plot")
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals")
        st.pyplot(fig)
    
    with col2:
        st.write("Actual vs Predicted Values")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=y_test, y=y_pred, ax=ax)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Actual vs Predicted Pollutant Values")
        st.pyplot(fig)

# ------------------------------------------------------------------
# Main Application
# ------------------------------------------------------------------
def main():
    configure_gemini()

    st.markdown("""
    <style>
    /* Main background color */
    .stApp {
        background: #272757;
        color: #ffffff;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #8686AC  !important;
    }
    
    /* Sidebar header text */
    [data-testid="stSidebar"] h1 {
        color: #e94560 !important;
    }
    
    /* Input fields */
    .stTextInput input, .stSlider {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border-color: #2d4263 !important;
    }
    
    /* Button styling */
    .stButton>button {
        background: #272757 !important;
        border: none !important;
        color: white !important;
        border-radius: 8px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🌍 Smart AQI Analyzer with AI Insights")
    st.markdown("""
    This application analyzes air quality data and provides AI-powered recommendations for improvement.
    Data is fetched from a public API and stored in Firebase Firestore for centralized access.
    """)
    
    # Sidebar inputs for state and city
    st.sidebar.header("Input Parameters")
    state = st.sidebar.text_input("State", "Delhi")
    city = st.sidebar.text_input("City", "Delhi")
    submit_button = st.sidebar.button("Analyze AQI Data")
    
    if submit_button:
        try:
            with st.spinner(f"Fetching AQI data for {city}, {state}..."):
                # Fetch data from the API
                raw_data = fetch_aqi_data(state=state, city=city, limit=5000)
                df_fetched = parse_aqi_data(raw_data)
                
                # Save each record to Firestore
                for _, row in df_fetched.iterrows():
                    record_data = row.to_dict()
                    save_aqi_data(city, state, record_data)
                    
            st.success("Data fetched and stored in Firebase!")
            
            # Retrieve all records for the specified city from Firestore
            all_city_records = get_aqi_data_by_city(city)
            st.write(f"Retrieved {len(all_city_records)} entries from Firebase for {city}.")
            
            # Convert the records to a DataFrame for analysis
            df = pd.DataFrame(all_city_records)
            
            # Perform EDA, AI insights, train model, and visualize results
            perform_eda(df)
            generate_ai_insights(df, city, state)
            model, X_test, y_test = train_model(df)
            visualize_results(model, X_test, y_test)
            
            # Option to download the dataset as CSV
            st.subheader("Download Data")
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="{city}_{state}_aqi_data.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please check your inputs and try again.")
    else:
        st.info("Enter state and city names, then click 'Analyze AQI Data' to begin analysis.")
        st.markdown("""
        ### Example Locations
        - Delhi, Delhi
        - Mumbai, Maharashtra
        - Kolkata, West Bengal
        - Chennai, Tamil Nadu
        - Bangalore, Karnataka
        """)

if __name__ == "__main__":
    main()
