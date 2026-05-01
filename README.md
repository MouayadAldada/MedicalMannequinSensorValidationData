# Multi-Modal Medical Mannequin - Comprehensive Sensor Validation Data



## Overview

This package contains comprehensive experimental validation data for a multi-modal medical training mannequin integrating 6 heterogeneous sensor types.

**Total Validation Samples: 6,900 across all sensors**

##  Complete Sensor Validation Results

### 1. Linear Potentiometer - Compression Depth (n=1,000)
- R² = 0.998, RMSE = ±1.79 mm
- Calibration: y = 1.047x - 0.010

### 2. Pressure Sensor (Abdominal) - Force Detection (n=1,500)
- RMSE = ±1.73 kPa
- Pain threshold detection: 100% accuracy
- Response: 38 ms

### 3. Temperature Control (n=800)
- Max error: ±0.94°C, Range: 32-40°C
- Response time: 90 seconds

### 4. Light Sensor (VEML7700) (n=1,200)
- RMSE: ±0.48 lux, Range: 50-1000 lux
- 16-bit resolution, Response: 85 ms

### 5. Airflow Sensor (n=1,000)
- Breath detection: 100% accuracy
- Threshold: -200 Pa, Response: 40 ms

### 6. Vision System (RPi Camera v3) (n=900)
- Tracking RMSE: ±1.52°
- 90% within ±2.5° spec, Response: 33 ms

### 7. Compression Rate (n=500)
- Mean accuracy: 96.1%
- Range: 80-140 CPM

##  Package Contents

**Data Files:** 7 CSV files (6,900 total samples)
**Analysis Scripts:** 3 Python scripts
**Validation Plots:** 7  figures 

**Contact:** Mouayad Aldada (ma3012@hw.ac.uk)  
Heriot-Watt University Dubai
