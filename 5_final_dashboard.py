import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

print("Creating Final Dashboard...")
os.makedirs('plots', exist_ok=True)

# Load existing images
try:
    img_pred = mpimg.imread('plots/8_predictor_forecast.png')
    img_sim = mpimg.imread('plots/10_wait_times_before_opt.png')
    img_opt = mpimg.imread('plots/12_optimization_results.png')
except FileNotFoundError as e:
    print(f"Error loading images: {e}. Ensure 2_predictor.py, 3_simulator.py, and 4_optimizer.py have been run.")
    exit(1)

# Create a master figure layout
fig = plt.figure(figsize=(20, 24))
fig.suptitle('Hospital Resource Allocation System: End-to-End Analytics', fontsize=32, fontweight='bold', y=0.98)

# 1. Predictor
ax1 = plt.subplot2grid((3, 1), (0, 0))
ax1.imshow(img_pred)
ax1.axis('off')
ax1.set_title("Phase 1: The Predictor (Time-Series Forecasting)", fontsize=24, pad=20)

# 2. Simulator
ax2 = plt.subplot2grid((3, 1), (1, 0))
ax2.imshow(img_sim)
ax2.axis('off')
ax2.set_title("Phase 2: The Simulator (Baseline Bottlenecks)", fontsize=24, pad=20)

# 3. Optimizer
ax3 = plt.subplot2grid((3, 1), (2, 0))
ax3.imshow(img_opt)
ax3.axis('off')
ax3.set_title("Phase 3: The Optimizer (Shift Planning & Staff Scaling)", fontsize=24, pad=20)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.subplots_adjust(hspace=0.3)
dashboard_path = 'plots/99_final_dashboard.png'
plt.savefig(dashboard_path, bbox_inches='tight', dpi=150)
print(f"Final dashboard saved to {dashboard_path}")
