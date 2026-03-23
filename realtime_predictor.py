
import pandas as pd
import time
from datetime import datetime
import random
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
import mysql.connector

warnings.filterwarnings('ignore')

# Default XAMPP MySQL Configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '', # Default XAMPP has no root password
}

DB_NAME = 'hospitalresource'

def setup_database():
    """Connects to XAMPP MySQL and creates the necessary database and tables."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Create database and table
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
        cursor.execute(f"USE {DB_NAME}")
        
        create_table_query = """
        CREATE TABLE IF NOT EXISTS weeklypredictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            emergency INT,
            surgery INT,
            general_medicine INT,
            ICU INT
        )
        """
        cursor.execute(create_table_query)
        conn.commit()
        cursor.close()
        conn.close()
        print(f"Database '{DB_NAME}' and table 'weeklypredictions' are ready.")
    except Exception as e:
        print(f"Error setting up database: {e}")
        print("Please ensure XAMPP MySQL Server is running.")
        exit(1)

def insert_predictions(predictions):
    """Inserts a single row of predictions into the MySQL database."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG, database=DB_NAME)
        cursor = conn.cursor()
        
        insert_query = """
        INSERT INTO weeklypredictions (emergency, surgery, general_medicine, ICU)
        VALUES (%s, %s, %s, %s)
        """
        
        data_tuple = (
            predictions.get('emergency', 0),
            predictions.get('surgery', 0),
            predictions.get('general_medicine', 0),
            predictions.get('ICU', 0)
        )
        
        cursor.execute(insert_query, data_tuple)
        conn.commit()
        cursor.close()
        conn.close()
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Successfully inserted new predictions into MySQL.")
    except Exception as e:
        print(f"Database insertion failed: {e}")

def run_realtime_predictor():
    print("Starting Real-Time Predictor Pipeline (Press Ctrl+C to stop)...")
    setup_database()
    
    # Load base historical data
    df = pd.read_csv('services_weekly.csv')
    services = df['service'].unique()
    
    current_week_target = 53
    
    while True:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        live_predictions = {}
        
        for s in services:
            service_df = df[df['service'] == s].sort_values('week')
            ts = service_df['patients_request'].values
            
            # To simulate a "live" changing environment for your Grafana dashboard, 
            # we introduce slight random variance (±10%) to the baseline trend before forecasting 
            # so the plot lines look realistic and dynamic.
            noise_factor = random.uniform(0.9, 1.1) 
            noisy_ts = ts * noise_factor
            
            model = ExponentialSmoothing(noisy_ts, trend='add', seasonal=None, initialization_method="heuristic")
            fit_model = model.fit()
            
            forecast = fit_model.forecast(1)[0]
            forecast = max(0, int(round(forecast)))
            
            live_predictions[s] = forecast
            
        insert_predictions(live_predictions)
        
        current_week_target += 1
        
        print("Sleeping for 10 seconds before next prediction wave...")
        time.sleep(10)

if __name__ == "__main__":
    run_realtime_predictor()
