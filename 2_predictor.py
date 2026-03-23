import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
import os

warnings.filterwarnings('ignore')

print("Starting Predictor Module...")
# Ensure plots dir exists
os.makedirs('plots', exist_ok=True)

# Load the weekly services data which contains "patients_request"
df = pd.read_csv('services_weekly.csv')

# We will forecast week 53 for each service using Exponential Smoothing
services = df['service'].unique()
forecasts = []

plt.figure(figsize=(15, 10))

for i, s in enumerate(services):
    service_df = df[df['service'] == s].sort_values('week')
    # Time series of patient requests
    ts = service_df['patients_request'].values
    weeks = service_df['week'].values
    
    # Fit Simple Exponential Smoothing or Holt's Linear Trend
    # Since we have 52 weeks, no clear strong seasonality specified but let's just use a simple Holt model
    model = ExponentialSmoothing(ts, trend='add', seasonal=None, initialization_method="heuristic")
    fit_model = model.fit()
    
    # Predict 1 step ahead (Week 53)
    forecast = fit_model.forecast(1)[0]
    # To be safe, round to integer and ensure non-negative
    forecast = max(0, int(round(forecast)))
    forecasts.append({'service': s, 'predicted_patients': forecast})
    
    print(f"Service: {s.ljust(18)} | Historical Avg: {int(ts.mean())} | Forecast for Week 53: {forecast}")
    
    # Plotting
    plt.subplot(2, 2, i+1)
    plt.plot(weeks, ts, label='Historical Inflow', marker='o')
    # Plot the predicted point
    plt.plot(53, forecast, marker='*', markersize=12, color='red', label='Forecast Wk 53')
    plt.title(f'{s.capitalize()} - Patient Inflow Forecast')
    plt.xlabel('Week')
    plt.ylabel('Patient Requests')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.savefig('plots/8_predictor_forecast.png')
print("Saved forecast plot to plots/8_predictor_forecast.png")

# Save predictions for the simulator
forecast_df = pd.DataFrame(forecasts)
forecast_df.to_csv('forecast_week_53.csv', index=False)
print("Saved forecasts to forecast_week_53.csv")
