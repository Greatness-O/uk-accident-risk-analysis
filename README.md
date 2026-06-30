# UK Road Accident Risk Analytics and Forecasting

This project analyses UK accident data from 2019. The findings will be used to outline recommendations for policy changes/interventions that should be made by the UK Government to improve road safety. A predictive model for accidents and injuries is also created.


## Overview

The project answers three practical road-safety questions:

1. **When do accidents concentrate?**  
   Accident frequency is analysed by hour, weekday and road-user group to identify commuter, school-run, weekend and night-time patterns.

2. **Where are the local hotspots?**  
   Hull, Humberside and East Riding accidents are mapped using Folium, heatmaps, marker clustering and geospatial clustering.

3. **Which conditions are linked to severity?**  
   Association-rule mining and a baseline severity classifier are used to explore how road type, light, weather, speed limit and urban/rural context relate to accident severity.

## Key features

- SQL extraction from the Department for Transport SQLite accident database
- Reproducible Python pipeline with reusable functions
- Time-based analysis of accidents by hour, weekday and road-user category
- Motorcycle risk analysis by engine capacity
- Pedestrian casualty analysis by hour, weekday and gender
- Apriori association-rule mining for accident severity
- KMeans and DBSCAN geospatial hotspot detection
- Interactive Folium map with marker clustering and heatmap layers
- SARIMAX forecasting for police-force accident counts
- Seasonal-naive baseline comparison for forecasting discipline
- Hull top-30 LSOA short-term daily accident forecast

## Repository structure

```
uk-road-accident-risk-analytics/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── raw/
│
├── notebooks/
│   └── 01_updated_accident_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   └── accident_pipeline.py
│
└── outputs/
    ├── figures/
    ├── maps/
    └── tables/
```

## Methods

### Temporal analysis

Accident records are parsed into date, hour, weekday, month and week features. Day-hour matrices are used to identify accident concentration during commuting periods, school-run windows and weekend/night-time periods.

### Geospatial clustering

The project includes two clustering approaches:

- **KMeans:** useful for producing fixed regional groupings and easy-to-interpret map clusters.
- **DBSCAN:** better suited to hotspot detection because it identifies dense local clusters and leaves isolated points as noise.

### Association-rule mining

Apriori is used to identify combinations of road, weather, light, surface, speed-limit and urban/rural conditions associated with accident severity. Rules are ranked by support, confidence and lift.

### Forecasting

SARIMAX is used to forecast weekly accident counts for selected police-force areas. A seasonal-naive baseline is included so model performance is not interpreted without a benchmark.

### Severity classification

A baseline Random Forest classifier is included to predict severity labels using road, time, environmental and location features. Evaluation uses balanced accuracy, macro F1 and weighted F1 to account for class imbalance.
