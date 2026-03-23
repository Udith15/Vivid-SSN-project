import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)
print("Starting Data Exploration and Analytics...")

# Load datasets
try:
    patients_df = pd.read_csv('patients.csv')
    services_df = pd.read_csv('services_weekly.csv')
    staff_df = pd.read_csv('staff.csv')
    schedule_df = pd.read_csv('staff_schedule.csv')
    print("Datasets loaded successfully.")
except Exception as e:
    print(f"Error loading datasets: {e}")
    exit(1)

# 1. Patients Data EDA
print("\n--- Patients Data ---")
print(patients_df.info())
patients_df['arrival_date'] = pd.to_datetime(patients_df['arrival_date'])
patients_df['departure_date'] = pd.to_datetime(patients_df['departure_date'])
patients_df['los'] = (patients_df['departure_date'] - patients_df['arrival_date']).dt.days

plt.figure(figsize=(10, 6))
sns.histplot(patients_df['age'], bins=20, kde=True, color='skyblue')
plt.title('Patient Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.savefig('plots/1_patient_age_distribution.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.countplot(data=patients_df, x='service', palette='Set2')
plt.title('Patients by Service Department')
plt.xlabel('Service')
plt.ylabel('Number of Patients')
plt.savefig('plots/2_patients_by_service.png')
plt.close()

# Inflow over time (monthly aggregation)
patients_df['arrival_month'] = patients_df['arrival_date'].dt.to_period('M')
monthly_inflow = patients_df.groupby('arrival_month').size()
plt.figure(figsize=(12, 6))
monthly_inflow.plot(kind='line', marker='o', color='coral')
plt.title('Monthly Patient Inflow (from patients.csv)')
plt.xlabel('Month')
plt.ylabel('Number of Arrivals')
plt.grid(True)
plt.savefig('plots/3_monthly_patient_inflow.png')
plt.close()
print("Saved patient analytics plots.")

# 2. Services Weekly EDA
print("\n--- Services Weekly Data ---")
print(services_df.info())

# Aggregate requests vs admitted by service
service_agg = services_df.groupby('service')[['patients_request', 'patients_admitted', 'patients_refused']].sum().reset_index()
service_agg_melt = service_agg.melt(id_vars='service', value_vars=['patients_admitted', 'patients_refused'], 
                                    var_name='Status', value_name='Count')
plt.figure(figsize=(10, 6))
sns.barplot(data=service_agg_melt, x='service', y='Count', hue='Status', palette='Pastel1')
plt.title('Total Patients Admitted vs Refused by Service')
plt.xlabel('Service')
plt.ylabel('Total Patients')
plt.savefig('plots/4_admitted_vs_refused_by_service.png')
plt.close()

# Trend of requests over weeks
plt.figure(figsize=(14, 6))
sns.lineplot(data=services_df, x='week', y='patients_request', hue='service', marker='o')
plt.title('Weekly Patient Requests by Service')
plt.xlabel('Week')
plt.ylabel('Number of Requests')
plt.grid(True)
plt.savefig('plots/5_weekly_requests_trend.png')
plt.close()
print("Saved service weekly analytics plots.")

# 3. Staff Data EDA
print("\n--- Staff Data ---")
print(staff_df.info())

plt.figure(figsize=(10, 6))
sns.countplot(data=staff_df, x='role', hue='service', palette='Dark2')
plt.title('Staff Role Distribution across Services')
plt.xlabel('Role')
plt.ylabel('Count')
plt.savefig('plots/6_staff_distribution.png')
plt.close()
print("Saved staff analytics plots.")

# 4. Schedule Data EDA
print("\n--- Staff Schedule Data ---")
# Count total staff present per week by service
schedule_agg = schedule_df.groupby(['week', 'service'])['present'].sum().reset_index()
plt.figure(figsize=(14, 6))
sns.lineplot(data=schedule_agg, x='week', y='present', hue='service', marker='o')
plt.title('Weekly Staff Availability by Service')
plt.xlabel('Week')
plt.ylabel('Total Staff Present')
plt.grid(True)
plt.savefig('plots/7_weekly_staff_availability.png')
plt.close()
print("Saved staff schedule analytics plots.")
print("Exploratory Data Analysis completed. Plots saved in 'plots' directory.")
