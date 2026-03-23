import pandas as pd
import simpy
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('plots', exist_ok=True)
print("Starting Optimizer...")

# Load Data
forecasts = pd.read_csv('forecast_week_53.csv')
schedule = pd.read_csv('staff_schedule.csv')

# Baseline for comparison
baseline_staff = schedule[schedule['week'] == 52].groupby('service')['present'].sum().to_dict()

SIM_TIME = 168  # 1 week = 168 hours
TARGET_WAIT_MINS = 20.0

SERVICE_PARAMS = {
    'emergency': (1.5, 0.3),
    'surgery': (4.0, 1.0),
    'general_medicine': (2.0, 0.5),
    'ICU': (24.0, 4.0)
}

def patient(env, service, staff_pool, wait_times):
    arrival_time = env.now
    with staff_pool.request() as req:
        yield req
        wait_times.append(env.now - arrival_time)
        mean, std = SERVICE_PARAMS.get(service, (2.0, 0.5))
        service_time = max(0.1, random.gauss(mean, std))
        yield env.timeout(service_time)

def patient_generator(env, service, staff_pool, total_patients, wait_times):
    if total_patients <= 0:
        return
    arrival_interval_mean = SIM_TIME / total_patients
    for i in range(total_patients):
        yield env.timeout(random.expovariate(1.0 / arrival_interval_mean))
        env.process(patient(env, service, staff_pool, wait_times))

def run_simulation(staff_config, seed=42):
    """Runs the simulation and returns 95th percentile wait times in minutes for each service."""
    random.seed(seed)
    env = simpy.Environment()
    staff_pools = {}
    wait_times_dict = {s: [] for s in staff_config.keys()}
    
    for _, row in forecasts.iterrows():
        s = row['service']
        num_patients = int(row['predicted_patients'])
        capacity = staff_config.get(s, 1)
        staff_pools[s] = simpy.Resource(env, capacity=capacity)
        env.process(patient_generator(env, s, staff_pools[s], num_patients, wait_times_dict[s]))
        
    env.run(until=SIM_TIME)
    
    results = {}
    for s, waits in wait_times_dict.items():
        if waits:
            p95 = np.percentile(waits, 95) * 60  # convert hours to minutes
        else:
            p95 = 0.0
        results[s] = p95
    return results

# Optimize staffing
optimal_staff = {}
optimized_waits = {}

for _, row in forecasts.iterrows():
    s = row['service']
    num_patients = int(row['predicted_patients'])
    
    if num_patients == 0:
        optimal_staff[s] = 1
        optimized_waits[s] = 0.0
        continue
        
    print(f"Optimizing {s} for {num_patients} forecasted patients...")
    # Start with a heuristic lower bound (Little's Law roughly: L = lambda * W => req_capacity = arrival_rate * mean_service_time)
    arrival_rate = num_patients / SIM_TIME
    mean_service, _ = SERVICE_PARAMS.get(s, (2.0, 0.5))
    min_staff = max(1, int(arrival_rate * mean_service))
    
    current_staff = min_staff
    while True:
        # We test this specific service isolation by running the whole hospital but only paying attention to this one.
        # Actually, running just the required service is faster. 
        # But our run_simulation function runs all. To isolate, we just pass the required config.
        # Let's mock the others to 1.
        test_config = {srv: 1 for srv in forecasts['service']}
        test_config[s] = current_staff
        
        waits = run_simulation(test_config, seed=42)
        p95 = waits[s]
        
        print(f"  Testing Staff={current_staff} -> P95 Wait: {p95:.2f} mins")
        if p95 <= TARGET_WAIT_MINS:
            optimal_staff[s] = current_staff
            optimized_waits[s] = p95
            break
        current_staff += 1
        # safety break
        if current_staff > 500:
            print("  Hit safety max limit of 500 staff.")
            optimal_staff[s] = current_staff
            optimized_waits[s] = p95
            break

print("\n--- Optimization Complete ---")
for s, cap in optimal_staff.items():
    print(f"Service: {s.ljust(18)} | Optimal Staff: {cap} | Achieving P95 Wait: {optimized_waits[s]:.2f} mins")

# Compare Baseline vs Optimal
# Get baseline wait times
baseline_config = {s: baseline_staff.get(s, 1) for s in baseline_staff}
baseline_waits = run_simulation(baseline_config, seed=42)

# Prepare DataFrame for saving the optimal plan
staff_plan_df = pd.DataFrame([
    {'service': s, 'optimal_staff_count': cap, 'predicted_patients': forecasts[forecasts['service']==s]['predicted_patients'].values[0]} 
    for s, cap in optimal_staff.items()
])
staff_plan_df.to_csv('optimal_staff_plan_week_53.csv', index=False)
print("Saved Optimal Staffing Shift Plan to optimal_staff_plan_week_53.csv")

# Visualization: Before vs After Wait Times
services = list(optimal_staff.keys())
b_waits = [baseline_waits.get(s, 0) for s in services]
o_waits = [optimized_waits.get(s, 0) for s in services]

x = np.arange(len(services))
width = 0.35

plt.figure(figsize=(14, 6))

# Plot 1: Wait Times
plt.subplot(1, 2, 1)
plt.bar(x - width/2, b_waits, width, label='Baseline Wait Time', color='salmon')
plt.bar(x + width/2, o_waits, width, label='Optimized Wait Time', color='lightgreen')
plt.axhline(TARGET_WAIT_MINS, color='r', linestyle='--', label='20 min Threshold')
plt.title('95th Percentile Wait Times (Before vs After)')
plt.xticks(x, services, rotation=15)
plt.ylabel('Minutes')
plt.legend()

# Plot 2: Staff Counts
b_staff = [baseline_staff.get(s, 1) for s in services]
o_staff = [optimal_staff.get(s, 1) for s in services]

plt.subplot(1, 2, 2)
plt.bar(x - width/2, b_staff, width, label='Baseline Staff Count', color='lightblue')
plt.bar(x + width/2, o_staff, width, label='Optimal Staff Count', color='gold')
plt.title('Staff Allocation (Before vs After)')
plt.xticks(x, services, rotation=15)
plt.ylabel('Number of Staff')
plt.legend()

plt.tight_layout()
plt.savefig('plots/12_optimization_results.png')
print("Saved Before vs After comparison plot to plots/12_optimization_results.png")
