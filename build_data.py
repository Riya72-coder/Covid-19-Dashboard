import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Essential: Use non-interactive backend for server deployment
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from prophet import Prophet
import os
import json
from datetime import datetime

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "images") # Save images to static folder for Flask
JSON_OUTPUT_PATH = os.path.join(BASE_DIR, 'dashboard_data.json')

# --- Dataset Configuration (Change these names if you add new files) ---
VACCINATION_FILE = "IndiaCovidVaccination2023.csv"
SENTIMENT_FILE = "COVID-19_Sentiments.csv"
MAIN_DATA_FILE = "output.csv"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Data Loading ---
def get_vaccination_df():
    """Loads the vaccination dataset."""
    vaccination_data_path = os.path.join(DATA_DIR, VACCINATION_FILE)
    if not os.path.exists(vaccination_data_path):
        raise FileNotFoundError(f"Vaccination file not found at {vaccination_data_path}.")
    
    return pd.read_csv(vaccination_data_path)

def get_sentiment_df():
    """Loads the sentiment dataset."""
    sentiment_data_path = os.path.join(DATA_DIR, SENTIMENT_FILE)
    if not os.path.exists(sentiment_data_path):
        print(f"Warning: Sentiment file not found at {sentiment_data_path}. Using placeholder data.")
        # Return a dummy dataframe so the app doesn't crash
        return pd.DataFrame({'sentiments/public opinion': [0], 'state_location': ['Unknown']})
    
    return pd.read_csv(sentiment_data_path)

def get_main_df():
    """Loads and preprocesses the main dataset."""
    analysis_data_path = os.path.join(DATA_DIR, MAIN_DATA_FILE)
    if not os.path.exists(analysis_data_path):
        raise FileNotFoundError(f"Analysis file not found at {analysis_data_path}. Please generate it first.")
    
    df = pd.read_csv(analysis_data_path)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['active_cases'] = df['total_cases'] - df['discharged'] - df['deaths']
    return df

# --- Analysis Functions ---

def get_kpis(df, vaccination_df, sentiment_df):
    """Calculates Key Performance Indicators."""
    latest_df = df.sort_values('date', ascending=False).drop_duplicates('state_location')
    
    def categorize_sentiment(score):
        if score > 0: return 'Positive'
        elif score < 0: return 'Negative'
        else: return 'Neutral'

    # Handle potential column name mismatches for sentiment
    sentiment_col = 'sentiments/public opinion'
    if sentiment_col not in sentiment_df.columns:
        # Try to find a similar column
        possible_cols = [c for c in sentiment_df.columns if 'sentiment' in c.lower() or 'opinion' in c.lower()]
        if possible_cols:
            sentiment_col = possible_cols[0]
        else:
            print(f"Warning: Sentiment column not found. Available: {sentiment_df.columns.tolist()}")
            sentiment_df[sentiment_col] = 0 # Default to 0 (Neutral) if missing

    sentiment_df['sentiment_category'] = sentiment_df[sentiment_col].apply(categorize_sentiment)
    overall_sentiment = sentiment_df['sentiment_category'].mode()[0]

    # Handle potential column name mismatches for vaccination doses
    doses_col = 'total_doses'
    if doses_col not in vaccination_df.columns:
        # Try to find a similar column containing 'total' and 'dose'
        possible_cols = [c for c in vaccination_df.columns if 'total' in c.lower() and 'dose' in c.lower()]
        if possible_cols:
            doses_col = possible_cols[0]
        else:
            print(f"Warning: 'total_doses' column not found. Available: {vaccination_df.columns.tolist()}")
            vaccination_df[doses_col] = 0 # Default to 0 if missing

    # Ensure the column is numeric (remove commas if present)
    if vaccination_df[doses_col].dtype == 'object':
        vaccination_df[doses_col] = pd.to_numeric(vaccination_df[doses_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

    kpis = {
        'total_cases': int(latest_df['total_cases'].sum()),
        'total_discharged': int(latest_df['discharged'].sum()),
        'total_deaths': int(latest_df['deaths'].sum()),
        'active_cases': int(latest_df['active_cases'].sum()),
        'total_vaccination_doses': int(vaccination_df[doses_col].sum()),
        'overall_sentiment': overall_sentiment,
        'last_updated': datetime.now().strftime("%B %d, %Y, %I:%M %p")
    }
    return kpis

def get_trends_over_time_data(df):
    """Generates data for 'Trends over Time' chart."""
    data_grouped_date = df.groupby('date')[['total_cases', 'discharged', 'deaths']].sum().reset_index()
    
    # Generate plot
    plt.figure(figsize=(12, 6))
    plt.plot(data_grouped_date['date'], data_grouped_date['total_cases'], label='Total Cases')
    plt.plot(data_grouped_date['date'], data_grouped_date['discharged'], label='Recoveries')
    plt.plot(data_grouped_date['date'], data_grouped_date['deaths'], label='Deaths')
    plt.title('Trends of COVID-19 Cases, Recoveries, and Deaths Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=12); plt.ylabel('Count', fontsize=12)
    plt.legend(loc='upper left'); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'trends_over_time.png'))
    plt.close()
    
    # Return data for Chart.js
    return {
        'labels': data_grouped_date['date'].dt.strftime('%Y-%m-%d').tolist(),
        'datasets': [
            {'label': 'Total Cases', 'data': data_grouped_date['total_cases'].tolist(), 'borderColor': '#36A2EB', 'fill': False},
            {'label': 'Recoveries', 'data': data_grouped_date['discharged'].tolist(), 'borderColor': '#4BC0C0', 'fill': False},
            {'label': 'Deaths', 'data': data_grouped_date['deaths'].tolist(), 'borderColor': '#FF6384', 'fill': False}
        ]
    }

def get_top_states_data(df):
    """Generates data for 'Top 10 States' bar chart and 'Top 5 States' pie chart."""
    state_cases = df.groupby('state_location')['total_cases'].sum().sort_values(ascending=False)
    
    # Bar chart data (Top 10)
    top_10_states = state_cases.head(10)
    plt.figure(figsize=(12, 6))
    top_10_states.plot(kind='bar', color='orange')
    plt.title('Top 10 States by Total Cases', fontsize=16)
    plt.xlabel('State', fontsize=12); plt.ylabel('Total Cases', fontsize=12)
    plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'top_10_states_cases.png'))
    plt.close()
    
    bar_chart_data = {
        'labels': top_10_states.index.tolist(),
        'datasets': [{'label': 'Total Cases', 'data': top_10_states.values.tolist(), 'backgroundColor': '#FF9F40'}]
    }

    # Pie chart data (Top 5)
    top_5_states = state_cases.head(5)
    plt.figure(figsize=(10, 10))
    top_5_states.plot.pie(autopct='%1.1f%%', startangle=140, colors=['#36A2EB', '#FF6384', '#4BC0C0', '#FFCE56', '#9966FF'])
    plt.title('Proportion of Total Cases by Top 5 States', fontsize=16)
    plt.ylabel(''); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'top_5_states_pie.png'))
    plt.close()

    pie_chart_data = {
        'labels': top_5_states.index.tolist(),
        'datasets': [{'data': top_5_states.values.tolist(), 'backgroundColor': ['#36A2EB', '#FF6384', '#4BC0C0', '#FFCE56', '#9966FF']}]
    }
    
    return bar_chart_data, pie_chart_data

def get_correlation_data(df):
    """Generates correlation heatmap."""
    # Only use columns that exist in the dataframe to prevent KeyError
    possible_cols = ['total_doses', 'dose1', 'dose_2', 'total_cases', 'discharged', 'deaths', 'population', 'discharge_ratio', 'death_ratio']
    numeric_cols = [c for c in possible_cols if c in df.columns]
    correlation_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Numeric Features', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'))
    plt.close()
    
    # Heatmap data is more complex for Chart.js, returning image path for now
    return 'images/correlation_heatmap.png'

def perform_forecasting(df):
    """Performs time series forecasting using ARIMA and Prophet."""
    print("Task 3: Performing time series forecasting...")
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.tz_localize(None)
    cases_ts = df.groupby('date')['total_cases'].sum()

    # ARIMA Model
    print("  > Fitting ARIMA model (this may take a few minutes)...")
    train = cases_ts[:int(0.8 * len(cases_ts))]
    test = cases_ts[int(0.8 * len(cases_ts)):]
    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(test))
    rmse = np.sqrt(mean_squared_error(test, predictions))
    print(f"ARIMA - Root Mean Squared Error: {rmse}")

    plt.figure(figsize=(12, 6))
    plt.plot(test.index, test, label="Actual Cases", color="blue")
    plt.plot(test.index, predictions, label="Predicted Cases (ARIMA)", color="red")
    plt.title("ARIMA: Actual vs Predicted COVID-19 Cases")
    plt.xlabel("Date"); plt.ylabel("Cases"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'arima_forecast.png'))
    plt.close()

    # Prophet Model
    print("  > Fitting Prophet model...")
    prophet_data = cases_ts.reset_index().rename(columns={'date': 'ds', 'total_cases': 'y'})
    prophet_model = Prophet()
    prophet_model.fit(prophet_data)
    future = prophet_model.make_future_dataframe(periods=30)
    forecast = prophet_model.predict(future)
    
    prophet_model.plot(forecast)
    plt.title("Prophet: Forecast of COVID-19 Cases")
    plt.xlabel("Date"); plt.ylabel("Total Cases")
    plt.savefig(os.path.join(OUTPUT_DIR, 'prophet_forecast.png'))
    plt.close()
    print("Forecast plots saved to static/images/ directory.")
    return 'images/arima_forecast.png', 'images/prophet_forecast.png'

def get_sentiment_distribution_data(df):
    """Generates data for 'Sentiment Distribution' chart."""
    def categorize_sentiment(score):
        if score > 0: return 'Positive'
        elif score < 0: return 'Negative'
        else: return 'Neutral'

    # Handle potential column name mismatches for sentiment
    sentiment_col = 'sentiments/public opinion'
    if sentiment_col not in df.columns:
        possible_cols = [c for c in df.columns if 'sentiment' in c.lower() or 'opinion' in c.lower()]
        if possible_cols:
            sentiment_col = possible_cols[0]
        else:
            df[sentiment_col] = 0 # Default to 0 if missing

    df['sentiment_category'] = df[sentiment_col].apply(categorize_sentiment)
    sentiment_counts = df['sentiment_category'].value_counts()
    
    # Sentiment Distribution Bar Chart
    plt.figure(figsize=(8, 5))
    sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment Category"); plt.ylabel("Frequency"); plt.xticks(rotation=0)
    plt.savefig(os.path.join(OUTPUT_DIR, 'sentiment_distribution.png'))
    plt.close()

    sentiment_dist_data = {
        'labels': sentiment_counts.index.tolist(),
        'datasets': [{'label': 'Frequency', 'data': sentiment_counts.values.tolist(), 'backgroundColor': ['#4BC0C0', '#FF6384', '#36A2EB']}]
    }
    
    return sentiment_dist_data


# --- Main Execution Function ---
def build_dashboard_data():
    print("Starting data build process...")
    
    # --- Safety Check: If data is missing (e.g. fresh GitHub deploy), create empty dashboard ---
    main_data_path = os.path.join(DATA_DIR, MAIN_DATA_FILE)
    if not os.path.exists(main_data_path):
        print(f"WARNING: Data file {MAIN_DATA_FILE} not found.")
        print("Generating placeholder data. Please upload the dataset via the Admin interface.")
        
        placeholder_data = {
            'kpi': {
                'total_cases': 0, 'total_discharged': 0, 'total_deaths': 0,
                'active_cases': 0, 'total_vaccination_doses': 0,
                'overall_sentiment': "No Data",
                'last_updated': "Please Upload Data"
            },
            'charts': {
                'trendsOverTime': {'labels': [], 'datasets': []},
                'topStatesBar': {'labels': [], 'datasets': []},
                'topStatesPie': {'labels': [], 'datasets': []},
                'sentimentDistribution': {'labels': [], 'datasets': []},
            },
            'image_paths': {
                'correlationHeatmap': '', 'arimaForecast': '', 'prophetForecast': ''
            }
        }
        with open(JSON_OUTPUT_PATH, 'w') as f:
            json.dump(placeholder_data, f, indent=4)
        return
    # -------------------------------------------------------------------------------------------

    main_df = get_main_df()
    vaccination_df = get_vaccination_df()
    sentiment_df = get_sentiment_df()
    print("Data loaded successfully.")
    
    # Generate all data and plots
    kpis = get_kpis(main_df.copy(), vaccination_df.copy(), sentiment_df.copy())
    print("KPIs generated.")
    trends_data = get_trends_over_time_data(main_df.copy())
    print("Trends data generated.")
    top_states_bar, top_states_pie = get_top_states_data(main_df.copy())
    print("Top states data generated.")
    correlation_img = get_correlation_data(main_df.copy())
    print("Correlation data generated.")
    arima_img, prophet_img = perform_forecasting(main_df.copy()) 
    print("Forecasting complete.")
    sentiment_dist_data = get_sentiment_distribution_data(sentiment_df.copy())
    print("Sentiment distribution data generated.")
    
    # --- Assemble final JSON object ---
    dashboard_data = {
        'kpi': kpis,
        'charts': {
            'trendsOverTime': trends_data,
            'topStatesBar': top_states_bar,
            'topStatesPie': top_states_pie,
            'sentimentDistribution': sentiment_dist_data,
        },
        'image_paths': {
            'correlationHeatmap': correlation_img,
            'arimaForecast': arima_img,
            'prophetForecast': prophet_img
        }
    }
    
    # Write to JSON file
    with open(JSON_OUTPUT_PATH, 'w') as f:
        json.dump(dashboard_data, f, indent=4)
        
    print(f"\nSuccessfully built data and saved to {JSON_OUTPUT_PATH}")

    print("Analysis complete. All plots saved in static/images/.")

# Allow running this script directly from terminal
if __name__ == "__main__":
    build_dashboard_data()
