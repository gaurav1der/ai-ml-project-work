# Will the Customer Accept the Coupon? - Data Analysis

A comprehensive analysis of customer behavior regarding coupon acceptance while driving, using machine learning and statistical techniques to identify key factors influencing acceptance rates.

## Project Overview

This project explores the question: "What factors determine whether a driver accepts a coupon delivered to their cell phone while driving?" Using survey data from Amazon Mechanical Turk, we analyze various demographic, contextual, and behavioral factors that influence coupon acceptance across different venue types.

## Jupyter notebook

[module5.ipynb](submission/module-5/module5.ipynb)

## Dataset Information

- **Source**: UCI Machine Learning Repository
- **Collection Method**: Amazon Mechanical Turk survey
- **Sample Size**: Multiple driving scenarios with various coupon types
- **Target Variable**: Y (1 = accepted, 0 = rejected)
- **Data File**: `data/coupons.csv`

### Coupon Types Analyzed
- Bar coupons
- Coffee House coupons  
- Restaurant coupons (< $20)
- Restaurant coupons ($20-$50)
- Carry out & Take away

### Data Attributes

**User Demographics:**
- Gender, Age, Marital Status, Children
- Education level, Occupation, Income
- Venue visit frequencies (bars, restaurants, coffee houses)

**Contextual Factors:**
- Driving destination (home, work, no urgent place)
- Weather conditions, Temperature, Time of day
- Passenger type (alone, partner, kids, friends)
- Location relative to coupon venue
- Distance and travel time to venue

## Analysis Structure

### 1. Exploratory Data Analysis
- Data cleaning and preprocessing
- Missing value handling (dropped 'car' column, forward-filled 'CarryAway')
- Overall acceptance rate: **57%**
- Distribution analysis of coupon types and contextual factors

### 2. Bar Coupon Analysis
Comprehensive investigation revealing:
- Frequency patterns vs. acceptance rates
- Age and demographic correlations
- Impact of passenger types and occupations
- Multi-factor condition analysis

### 3. Coffee House Coupon Deep Dive

**Key Findings:**
- **Overall acceptance rate: 50%**
- **High-value segments identified:**
  - Frequent visitors (>3 times/month): **67.26% acceptance**
  - Drivers under 21: **67.8% acceptance**
  - Healthcare practitioners: **76.0% acceptance**
  - Students: **61.4% acceptance**

## Major Insights

### Coffee House Coupon Acceptance Drivers

**Demographics & Behavior:**
- **Age**: Under 21 years (67.8% vs. other age groups)
- **Frequency**: Regular coffee house visitors (67.26% vs. 44.59%)
- **Occupation**: Healthcare practitioners (76.0%), Students (61.4%)
- **Income**: Lower to middle-income brackets show higher acceptance

**Contextual Factors:**
- **Social**: Friends (59.7%) or Partner (56.7%) vs. Alone (43.3%)
- **Destination**: No urgent place (57.8%) vs. Home (36.2%) or Work (44.0%)
- **Timing**: 10AM optimal (63.4% acceptance rate)
- **Weather**: Warmer temperatures (80°F: 52.6% vs. 30°F: 44.1%)
- **Location**: Same direction as destination (52.65% vs. 48.94%)
- **Distance**: <25 minutes travel time (significant drop-off at 25+ minutes: 34.23%)

## Business Recommendations

### Targeting Strategy
1. **Primary Targets**: Frequent coffee house visitors, under-21 demographic
2. **Professional Focus**: Healthcare workers and students
3. **Social Context**: Target drivers with companions over solo drivers

### Optimal Delivery Conditions
1. **Timing**: Late morning hours (around 10AM)
2. **Weather**: Warmer days (80°F optimal)
3. **Location**: Same direction as destination, <25 minutes away
4. **Context**: Non-urgent trips, social passengers

### Expected Impact
- **67.26%** acceptance rate for frequent visitors vs. 44.59% baseline
- **23% higher** acceptance with optimal demographic targeting
- **Distance optimization** can prevent 30%+ acceptance rate drop

## Technical Implementation

### Dependencies
```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
```

### Usage
```bash
# Run the analysis
jupyter notebook module5.ipynb
```

### Data Preprocessing
- Removed 'car' column (high missing values)
- Forward-filled 'CarryAway' missing values
- Age categorization and numerical conversion for analysis
- Frequency categorization for behavioral analysis

## Files Structure
```
├── module5.ipynb          # Main analysis notebook
├── data/
│   └── coupons.csv       # Source dataset
└── README.md             # This file
```

## Key Metrics Summary

| Segment | Acceptance Rate | Improvement vs. Baseline |
|---------|----------------|-------------------------|
| Overall Coffee House | 50.0% | - |
| Frequent Visitors (>3/month) | 67.26% | +17.26% |
| Under 21 Age Group | 67.8% | +17.8% |
| Healthcare Practitioners | 76.0% | +26.0% |
| With Friends | 59.7% | +9.7% |
| 10AM Timing | 63.4% | +13.4% |
| Same Direction | 52.65% | +2.65% |

## Future Work

1. **Predictive Modeling**: Build machine learning models for acceptance prediction
2. **A/B Testing Framework**: Design experiments for coupon optimization
3. **Real-time Personalization**: Implement dynamic coupon targeting
4. **Cross-venue Analysis**: Compare patterns across all coupon types
5. **Seasonal Analysis**: Incorporate temporal trends and seasonality

---

*This analysis provides actionable insights for targeted marketing strategies and personalized coupon delivery systems.*