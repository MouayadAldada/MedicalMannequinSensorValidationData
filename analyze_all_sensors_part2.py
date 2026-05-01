

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import os

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

os.makedirs('validation_plots', exist_ok=True)


# 4. LIGHT SENSOR (VEML7700) - Ambient Illuminance

print("\n" + "="*80)
print("4. LIGHT SENSOR (VEML7700) - Ambient Illuminance Analysis")
print("="*80)

df_light = pd.read_csv('light_sensor_validation.csv')

light_rmse = np.sqrt(mean_squared_error(df_light['Actual_Illuminance_lux'], 
                                         df_light['Measured_Illuminance_lux']))
mean_response_light = df_light['Response_Time_ms'].mean()

print(f"\nLight Sensor Statistics (n={len(df_light):,}):")
print(f"  Range:                 50-1000 lux")
print(f"  Resolution:            16-bit (0.004 lux)")
print(f"  RMSE:                  ±{light_rmse:.2f} lux")
print(f"  Mean Response Time:    {mean_response_light:.1f} ms")
print(f"  Pupil Size Range:      {df_light['Pupil_Size_mm'].min():.1f}-{df_light['Pupil_Size_mm'].max():.1f} mm")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Illuminance calibration
axes[0,0].scatter(df_light['Actual_Illuminance_lux'], 
                  df_light['Measured_Illuminance_lux'], alpha=0.3, s=20)
axes[0,0].plot([50, 1000], [50, 1000], 'r--', linewidth=2, label='Perfect agreement')
axes[0,0].axvline(1000, color='orange', linestyle=':', linewidth=2, label='Constriction threshold')
axes[0,0].axvline(50, color='green', linestyle=':', linewidth=2, label='Dilation threshold')
axes[0,0].set_xlabel('Actual Illuminance (lux)', fontsize=12)
axes[0,0].set_ylabel('Measured Illuminance (lux)', fontsize=12)
axes[0,0].set_title(f'Light Sensor Calibration (RMSE = ±{light_rmse:.2f} lux)', fontsize=14, fontweight='bold')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Plot 2: Pupil response to light
axes[0,1].scatter(df_light['Measured_Illuminance_lux'], 
                  df_light['Pupil_Size_mm'], alpha=0.3, s=20, c=df_light['Pupil_Size_mm'], cmap='viridis')
axes[0,1].set_xlabel('Illuminance (lux)', fontsize=12)
axes[0,1].set_ylabel('Pupil Size (mm)', fontsize=12)
axes[0,1].set_title('Pupillary Light Reflex Response', fontsize=14, fontweight='bold')
axes[0,1].axhline(2, color='r', linestyle=':', linewidth=1.5, label='Min (constricted)')
axes[0,1].axhline(8, color='b', linestyle=':', linewidth=1.5, label='Max (dilated)')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Plot 3: Response time distribution
axes[1,0].hist(df_light['Response_Time_ms'], bins=40, edgecolor='black', alpha=0.7)
axes[1,0].axvline(85, color='r', linestyle='--', linewidth=2, label='Specification: 85 ms')
axes[1,0].set_xlabel('Response Time (ms)', fontsize=12)
axes[1,0].set_ylabel('Frequency', fontsize=12)
axes[1,0].set_title(f'Response Time Distribution (Mean = {mean_response_light:.1f} ms)', fontsize=14, fontweight='bold')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Plot 4: Error distribution
light_errors = df_light['Measured_Illuminance_lux'] - df_light['Actual_Illuminance_lux']
axes[1,1].hist(light_errors, bins=50, edgecolor='black', alpha=0.7)
axes[1,1].axvline(0, color='r', linestyle='--', linewidth=2)
axes[1,1].set_xlabel('Measurement Error (lux)', fontsize=12)
axes[1,1].set_ylabel('Frequency', fontsize=12)
axes[1,1].set_title('Light Sensor Error Distribution', fontsize=14, fontweight='bold')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('validation_plots/4_light_sensor_analysis.png', dpi=300, bbox_inches='tight')
print("   Saved: validation_plots/4_light_sensor_analysis.png")


# 5. PRESSURE SENSOR (Airflow) - Breath Detection

print("\n" + "="*80)
print("5. PRESSURE SENSOR (Airflow) - Breath Detection Analysis")
print("="*80)

df_airflow = pd.read_csv('airflow_sensor_validation.csv')

breath_accuracy = (df_airflow['Breath_Detected'] == (df_airflow['Measured_Pressure_Pa'] < -200).astype(int)).mean() * 100
mean_response_airflow = df_airflow['Response_Time_ms'].mean()

print(f"\nAirflow Sensor Statistics (n={len(df_airflow):,}):")
print(f"  Detection Threshold:   -200 Pa")
print(f"  Breath Detection Accuracy: {breath_accuracy:.1f}%")
print(f"  Mean Response Time:    {mean_response_airflow:.1f} ms")
print(f"  Response Time Range:   {df_airflow['Response_Time_ms'].min():.1f}-{df_airflow['Response_Time_ms'].max():.1f} ms")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Pressure measurements
axes[0,0].scatter(df_airflow['Actual_Pressure_Pa'], 
                  df_airflow['Measured_Pressure_Pa'], alpha=0.3, s=20)
axes[0,0].plot([-400, 50], [-400, 50], 'r--', linewidth=2, label='Perfect agreement')
axes[0,0].axhline(-200, color='orange', linestyle=':', linewidth=2, label='Breath threshold')
axes[0,0].axvline(-200, color='orange', linestyle=':', linewidth=2)
axes[0,0].set_xlabel('Actual Pressure (Pa)', fontsize=12)
axes[0,0].set_ylabel('Measured Pressure (Pa)', fontsize=12)
axes[0,0].set_title('Airflow Pressure Sensor Calibration', fontsize=14, fontweight='bold')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Plot 2: Breath detection zones
breath_yes = df_airflow[df_airflow['Breath_Detected'] == 1]['Measured_Pressure_Pa']
breath_no = df_airflow[df_airflow['Breath_Detected'] == 0]['Measured_Pressure_Pa']
axes[0,1].hist([breath_yes, breath_no], bins=30, label=['Breath Detected', 'No Breath'], 
               edgecolor='black', alpha=0.7)
axes[0,1].axvline(-200, color='r', linestyle='--', linewidth=2, label='Threshold')
axes[0,1].set_xlabel('Measured Pressure (Pa)', fontsize=12)
axes[0,1].set_ylabel('Frequency', fontsize=12)
axes[0,1].set_title(f'Breath Detection Distribution (Accuracy = {breath_accuracy:.1f}%)', fontsize=14, fontweight='bold')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Plot 3: Response time
axes[1,0].hist(df_airflow['Response_Time_ms'], bins=30, edgecolor='black', alpha=0.7)
axes[1,0].axvline(50, color='r', linestyle='--', linewidth=2, label='Max specification: 50 ms')
axes[1,0].set_xlabel('Response Time (ms)', fontsize=12)
axes[1,0].set_ylabel('Frequency', fontsize=12)
axes[1,0].set_title(f'Response Time Distribution (Mean = {mean_response_airflow:.1f} ms)', fontsize=14, fontweight='bold')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Plot 4: Airflow pressure pattern (sample data)
sample_data = df_airflow.head(200)
axes[1,1].plot(sample_data['Trial_Number'], sample_data['Measured_Pressure_Pa'], linewidth=1.5)
axes[1,1].axhline(-200, color='r', linestyle='--', linewidth=2, label='Breath threshold')
axes[1,1].fill_between(sample_data['Trial_Number'], -400, -200, alpha=0.2, color='green', label='Breath zone')
axes[1,1].set_xlabel('Sample Number', fontsize=12)
axes[1,1].set_ylabel('Pressure (Pa)', fontsize=12)
axes[1,1].set_title('Airflow Pressure Pattern (Sample Data)', fontsize=14, fontweight='bold')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('validation_plots/5_airflow_sensor_analysis.png', dpi=300, bbox_inches='tight')
print("   Saved: validation_plots/5_airflow_sensor_analysis.png")


# 6. VISION SYSTEM (RPi Camera v3) - Gaze Tracking

print("\n" + "="*80)
print("6. VISION SYSTEM (RPi Camera v3) - Gaze Tracking Analysis")
print("="*80)

df_vision = pd.read_csv('vision_system_validation.csv')

vision_rmse = np.sqrt(mean_squared_error(df_vision['Actual_Gaze_Angle_deg'], 
                                          df_vision['Measured_Gaze_Angle_deg']))
mean_error = df_vision['Tracking_Error_deg'].mean()
mean_response_vision = df_vision['Response_Time_ms'].mean()

print(f"\nVision System Statistics (n={len(df_vision):,}):")
print(f"  Tracking RMSE:         ±{vision_rmse:.2f}°")
print(f"  Mean Tracking Error:   ±{mean_error:.2f}°")
print(f"  Mean Response Time:    {mean_response_vision:.1f} ms")
print(f"  Frame Rate:            30 fps")
print(f"  Errors within ±2.5°:   {(df_vision['Tracking_Error_deg'] <= 2.5).sum() / len(df_vision) * 100:.1f}%")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Gaze tracking accuracy
axes[0,0].scatter(df_vision['Actual_Gaze_Angle_deg'], 
                  df_vision['Measured_Gaze_Angle_deg'], alpha=0.3, s=20)
axes[0,0].plot([-30, 30], [-30, 30], 'r--', linewidth=2, label='Perfect tracking')
axes[0,0].fill_between([-30, 30], [-32.5, 27.5], [-27.5, 32.5], alpha=0.2, 
                        color='green', label='±2.5° tolerance')
axes[0,0].set_xlabel('Actual Gaze Angle (°)', fontsize=12)
axes[0,0].set_ylabel('Measured Gaze Angle (°)', fontsize=12)
axes[0,0].set_title(f'Gaze Tracking Accuracy (RMSE = ±{vision_rmse:.2f}°)', fontsize=14, fontweight='bold')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Plot 2: Tracking error distribution
axes[0,1].hist(df_vision['Tracking_Error_deg'], bins=50, edgecolor='black', alpha=0.7)
axes[0,1].axvline(2.5, color='r', linestyle='--', linewidth=2, label='Specification: ±2.5°')
axes[0,1].set_xlabel('Tracking Error (°)', fontsize=12)
axes[0,1].set_ylabel('Frequency', fontsize=12)
axes[0,1].set_title(f'Tracking Error Distribution (Mean = ±{mean_error:.2f}°)', fontsize=14, fontweight='bold')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Plot 3: Response time
axes[1,0].hist(df_vision['Response_Time_ms'], bins=40, edgecolor='black', alpha=0.7)
axes[1,0].axvline(33, color='r', linestyle='--', linewidth=2, label='Specification: 33 ms (30 fps)')
axes[1,0].set_xlabel('Response Time (ms)', fontsize=12)
axes[1,0].set_ylabel('Frequency', fontsize=12)
axes[1,0].set_title(f'Response Time Distribution (Mean = {mean_response_vision:.1f} ms)', fontsize=14, fontweight='bold')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Plot 4: Tracking error vs angle
axes[1,1].scatter(df_vision['Actual_Gaze_Angle_deg'], 
                  df_vision['Tracking_Error_deg'], alpha=0.3, s=20)
axes[1,1].axhline(2.5, color='r', linestyle='--', linewidth=2, label='±2.5° spec')
axes[1,1].set_xlabel('Actual Gaze Angle (°)', fontsize=12)
axes[1,1].set_ylabel('Tracking Error (°)', fontsize=12)
axes[1,1].set_title('Tracking Error vs Gaze Angle', fontsize=14, fontweight='bold')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('validation_plots/6_vision_system_analysis.png', dpi=300, bbox_inches='tight')
print("   Saved: validation_plots/6_vision_system_analysis.png")


# 7. COMPRESSION RATE VALIDATION

print("\n" + "="*80)
print("7. COMPRESSION RATE VALIDATION")
print("="*80)

df_rate = pd.read_csv('compression_rate_validation.csv')

mean_accuracy = df_rate['Accuracy_Percent'].mean()
std_accuracy = df_rate['Accuracy_Percent'].std()

print(f"\nCompression Rate Statistics (n={len(df_rate):,}):")
print(f"  Rate Range:            80-140 CPM")
print(f"  Mean Accuracy:         {mean_accuracy:.1f}%")
print(f"  Std Deviation:         {std_accuracy:.2f}%")
print(f"  Accuracy within ±5%:   {((df_rate['Accuracy_Percent'] >= 91) & (df_rate['Accuracy_Percent'] <= 101)).sum() / len(df_rate) * 100:.1f}%")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Rate accuracy
axes[0,0].scatter(df_rate['Target_Rate_CPM'], df_rate['Measured_Rate_CPM'], alpha=0.3, s=30)
axes[0,0].plot([80, 140], [80, 140], 'r--', linewidth=2, label='Perfect accuracy')
axes[0,0].fill_between([80, 140], [76, 133], [84, 147], alpha=0.2, color='green', label='±5% tolerance')
axes[0,0].axhline(100, color='orange', linestyle=':', linewidth=1.5, label='AHA guideline min')
axes[0,0].axhline(120, color='orange', linestyle=':', linewidth=1.5, label='AHA guideline max')
axes[0,0].set_xlabel('Target Rate (CPM)', fontsize=12)
axes[0,0].set_ylabel('Measured Rate (CPM)', fontsize=12)
axes[0,0].set_title(f'Compression Rate Accuracy (Mean = {mean_accuracy:.1f}%)', fontsize=14, fontweight='bold')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Plot 2: Accuracy distribution
axes[0,1].hist(df_rate['Accuracy_Percent'], bins=40, edgecolor='black', alpha=0.7)
axes[0,1].axvline(96, color='r', linestyle='--', linewidth=2, label='Target: 96%')
axes[0,1].set_xlabel('Accuracy (%)', fontsize=12)
axes[0,1].set_ylabel('Frequency', fontsize=12)
axes[0,1].set_title('Compression Rate Accuracy Distribution', fontsize=14, fontweight='bold')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Plot 3: Error vs target rate
rate_errors = df_rate['Measured_Rate_CPM'] - df_rate['Target_Rate_CPM']
axes[1,0].scatter(df_rate['Target_Rate_CPM'], rate_errors, alpha=0.3, s=30)
axes[1,0].axhline(0, color='r', linestyle='--', linewidth=2)
axes[1,0].set_xlabel('Target Rate (CPM)', fontsize=12)
axes[1,0].set_ylabel('Error (CPM)', fontsize=12)
axes[1,0].set_title('Rate Measurement Error vs Target', fontsize=14, fontweight='bold')
axes[1,0].grid(True, alpha=0.3)

# Plot 4: Accuracy by rate zone
rate_bins = pd.cut(df_rate['Target_Rate_CPM'], bins=[80, 100, 120, 140], labels=['80-100', '100-120', '120-140'])
df_rate['Rate_Zone'] = rate_bins
zone_accuracy = df_rate.groupby('Rate_Zone')['Accuracy_Percent'].mean()
axes[1,1].bar(range(len(zone_accuracy)), zone_accuracy.values, edgecolor='black', alpha=0.7)
axes[1,1].set_xticks(range(len(zone_accuracy)))
axes[1,1].set_xticklabels(zone_accuracy.index)
axes[1,1].axhline(96, color='r', linestyle='--', linewidth=2, label='Overall mean')
axes[1,1].set_xlabel('Rate Zone (CPM)', fontsize=12)
axes[1,1].set_ylabel('Mean Accuracy (%)', fontsize=12)
axes[1,1].set_title('Accuracy by Rate Zone', fontsize=14, fontweight='bold')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('validation_plots/7_compression_rate_analysis.png', dpi=300, bbox_inches='tight')
print("   Saved: validation_plots/7_compression_rate_analysis.png")

print("\n" + "="*80)
print("ALL SENSOR PLOTS GENERATED SUCCESSFULLY")
print("="*80)
