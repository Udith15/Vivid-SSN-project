import pandas as pd
import simpy
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('plots', exist_ok=True)
print("Starting Simulator...")

# Load Data
forecasts = pd.read_csv('forecast_week_53.csv')
schedule = pd.read_csv('staff_schedule.csv')
staff = pd.read_csv('staff.csv')

# Use Week 52 as the baseline staffing config
current_week = 52
schedule_w52 = schedule[schedule['week'] == current_week]
baseline_staff = schedule_w52.groupby(['service'])['present'].sum().reset_index()

# Simulation Parameters
SIM_TIME = 168  # 1 week = 168 hours

# Service time config (mean_hours, std_hours)
SERVICE_PARAMS = {
    'emergency': (1.5, 0.3),
    'surgery': (4.0, 1.0),
    'general_medicine': (2.0, 0.5),
    'ICU': (24.0, 4.0)
}

wait_times = []

def patient(env, name, service, staff_pool):
    arrival_time = env.now
    with staff_pool.request() as req:
        yield req
        wait_time = env.now - arrival_time
        wait_times.append({'service': service, 'wait_time_hours': wait_time})
        
        # Determine service duration
        mean, std = SERVICE_PARAMS.get(service, (2.0, 0.5))
        service_time = max(0.1, random.gauss(mean, std))
        yield env.timeout(service_time)

def patient_generator(env, service, staff_pool, total_patients):
    if total_patients <= 0:
        return
    arrival_interval_mean = SIM_TIME / total_patients
    for i in range(total_patients):
        yield env.timeout(random.expovariate(1.0 / arrival_interval_mean))
        env.process(patient(env, f'Patient_{i}_{service}', service, staff_pool))

env = simpy.Environment()
staff_pools = {}

print("--- Simulation Configuration ---")
for _, row in forecasts.iterrows():
    s = row['service']
    num_patients = int(row['predicted_patients'])
    
    # Capacity is the total staff present for that service in Week 52
    cap_row = baseline_staff[baseline_staff['service'] == s]['present']
    capacity = cap_row.values[0] if not cap_row.empty else 1
    if capacity == 0:
        capacity = 1
        
    print(f"Service: {s.ljust(18)} | Staff Capacity: {capacity} | Forecasted Patients: {num_patients}")
    
    staff_pools[s] = simpy.Resource(env, capacity=int(capacity))
    env.process(patient_generator(env, s, staff_pools[s], num_patients))

print("Running simulation...")
env.run(until=SIM_TIME)

# Analytics & Visualization
df_wait = pd.DataFrame(wait_times)
df_wait['wait_time_mins'] = df_wait['wait_time_hours'] * 60

print("\n--- Current Bottlenecks (Baseline Wait Times) ---")
for s in df_wait['service'].unique():
    s_wait = df_wait[df_wait['service'] == s]['wait_time_mins']
    if not s_wait.empty:
        p95 = np.percentile(s_wait, 95)
        avg = s_wait.mean()
        print(f"{s.ljust(18)} - 95th Percentile: {p95:7.2f} mins | Avg Wait: {avg:7.2f} mins")

# Ensure saving works even if there's no data
if not df_wait.empty:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_wait, x='service', y='wait_time_mins', palette='Set3')
    plt.axhline(20, color='r', linestyle='--', label='20 min Threshold')
    plt.title('Patient Wait Times - Baseline Staffing Configuration')
    plt.ylabel('Wait Time (minutes)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/10_wait_times_before_opt.png')
    print("\nSaved wait time plot to plots/10_wait_times_before_opt.png")
