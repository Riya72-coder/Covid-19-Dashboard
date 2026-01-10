from flask import Flask, render_template, request, jsonify
import joblib
import os
import json
import matplotlib
matplotlib.use('Agg') # Essential: Use non-interactive backend for server
import matplotlib.pyplot as plt
import io
import base64
import threading
# Import the build function from your script
from build_data import build_dashboard_data, DATA_DIR, MAIN_DATA_FILE, SENTIMENT_FILE

app = Flask(__name__)

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_DATA_PATH = os.path.join(BASE_DIR, "dashboard_data.json")
MODEL_PATH = os.path.join(BASE_DIR, "covid_model.pkl")

# --- Load Model Globally (Optimization) ---
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None

# --- Route for the dashboard ---
@app.route('/dashboard')
def dashboard():
    """
    Loads pre-computed analysis data from a JSON file and renders the dashboard.
    """
    try:
        with open(JSON_DATA_PATH, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        return "Error: The `dashboard_data.json` file was not found. Please run the `build_data.py` script first.", 500

    # Pass the entire data object to the template
    return render_template('dashboard.html', data=data)

@app.route('/')
@app.route('/info')
def info():
    """Renders the Covid-19 Info page."""
    return render_template('covid-info-dashboard.html')

# --- Route for prediction (to be updated later) ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles prediction requests. This is a placeholder and will be updated.
    """
    if model is None:
        return jsonify(error="Model file not found. Please ensure covid_model.pkl exists."), 404

    # Get user input from form
    try:
        user_input = float(request.form['user_input'])
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Invalid user input."), 400

    # Make prediction using the model
    prediction = model.predict([[user_input]])

    # --- Generate Live Graph for this Prediction ---
    plt.figure(figsize=(6, 4))
    # Create a bar chart showing the predicted value
    plt.bar(['Predicted Cases'], [prediction[0]], color='#0d6efd', width=0.4)
    plt.title(f'Model Prediction for Input: {user_input}')
    plt.ylabel('Cases')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save plot to a temporary buffer (memory) instead of a file
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close() # Close the plot to free memory

    # Return the prediction AND the graph URL
    return jsonify(prediction=prediction[0], plot_url=plot_url)

# --- Route for Updating Data (Admin Feature) ---
@app.route('/update_data', methods=['GET', 'POST'])
def update_data():
    """
    Allows the admin to upload a new CSV file and trigger a dashboard refresh.
    """
    if request.method == 'POST':
        # Check if file is present
        if 'file' not in request.files:
            return "No file part", 400
        
        file = request.files['file']
        
        if file.filename == '':
            return "No selected file", 400

        if file:
            # Determine which file to save based on user selection
            file_type = request.form.get('file_type')
            if file_type == 'sentiment':
                save_path = os.path.join(DATA_DIR, SENTIMENT_FILE)
                print("Uploading Sentiment File...")
            else:
                save_path = os.path.join(DATA_DIR, MAIN_DATA_FILE)
                print("Uploading Main Dataset...")
            
            file.save(save_path)
            
            # Run the heavy analysis in a background thread so the browser doesn't freeze
            thread = threading.Thread(target=build_dashboard_data)
            thread.start()
            
            return """
            <div style="text-align:center; padding: 50px; font-family: sans-serif;">
                <h1 style="color: green;">Update Started!</h1>
                <p>The file has been uploaded successfully.</p>
                <p>The analysis script (ARIMA/Prophet) is running in the background.</p>
                <p>Please wait <b>3-5 minutes</b> for the charts to refresh, then <a href="/dashboard">Go to Dashboard</a>.</p>
            </div>
            """

    # Render a simple upload form
    return """
    <form method="post" enctype="multipart/form-data" style="text-align:center; padding: 50px; font-family: sans-serif;">
        <h1>Update Dashboard Data</h1>
        <p>Select the file type and upload the CSV.</p>
        <select name="file_type" style="padding: 10px; margin-bottom: 15px;">
            <option value="main">Main Dataset (output.csv)</option>
            <option value="sentiment">Sentiment Data (COVID-19_Sentiments.csv)</option>
        </select>
        <br>
        <input type="file" name="file" accept=".csv" required>
        <br><br>
        <button type="submit" style="padding: 10px 20px; background: #0d6efd; color: white; border: none; border-radius: 5px; cursor: pointer;">Upload & Refresh</button>
    </form>
    """

if __name__ == '__main__':
    app.run(debug=True)