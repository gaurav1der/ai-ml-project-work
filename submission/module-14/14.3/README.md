# 🚗 Daily Work Commute Decision Tree

## 📋 Overview

This project transforms the daily decision of "How should I get to work today?" into a visual decision tree and validates it using machine learning. The decision tree considers three key factors that everyone evaluates when choosing their morning commute: weather conditions, available time, and traffic levels.

---

## 🌳 Decision Tree Structure

- **Root Node (Level 1):** How to get to work today?
- **Level 2:** Weather conditions (primary safety factor)
- **Level 3:** Time availability (urgency consideration)
- **Level 4:** Traffic level (efficiency optimization)

---

## 🎨 Decision Factors

### Primary Inputs:
- **🌤️ Weather Conditions:**
  - 1 = Sunny
  - 2 = Cloudy  
  - 3 = Rainy
  - 4 = Stormy

- **⏰ Time Available:**
  - 1 = Rushed (running late)
  - 2 = Plenty of Time (comfortable schedule)

- **🚦 Traffic Level:**
  - 1 = Light Traffic
  - 3 = Heavy Traffic

---

## 🚗 Transportation Options

The decision tree recommends from these commute methods:
- **🚗 Drive** - Personal vehicle (fastest, most control)
- **🚌 Bus** - Public transit (avoids parking/traffic stress)
- **🚶 Walk** - On foot (healthy, weather-dependent)
- **🚴 Bike** - Bicycle (eco-friendly, moderate speed)
- **🏠 Work from Home** - Remote work (safety priority)

---

## 🧠 Decision Logic

### Safety First Approach:
1. **Stormy Weather (≥ 3.5)** → Work from Home
   - Prioritizes safety over attendance

### Time-Based Decisions:
2. **Good Weather + Rushed** → Drive
   - Speed and reliability when running late

3. **Good Weather + Plenty of Time** → Consider traffic:
   - Light traffic → Walk/Bike (healthy options)
   - Heavy traffic → Bus (stress-free alternative)

---

## 🤖 Machine Learning Implementation

### Technical Details
- **Algorithm:** DecisionTreeClassifier (scikit-learn)
- **Dataset:** 200 synthetic commute scenarios
- **Max Depth:** 3 levels
- **Accuracy:** 100% (perfect logic validation)

### Model Performance
| Metric | Value |
|--------|-------|
| **Accuracy** | 100% |
| **Dataset Size** | 200 samples |
| **Tree Depth** | 3 levels |
| **Transportation Classes** | 5 options |
| **Random State** | 42 (reproducible) |

---

## 📊 Feature Importance

Based on the trained model:

1. **Weather** - Primary decision factor (safety consideration)
2. **Time Available** - Secondary factor (urgency level)
3. **Traffic Level** - Tertiary factor (efficiency optimization)

*Weather emerges as the most important feature, confirming that safety considerations drive commute decisions.*

---

## 🔮 Example Predictions

| Weather | Time | Traffic | Scenario | Recommendation |
|---------|------|---------|----------|----------------|
| Sunny | Rushed | Light | Perfect driving conditions | **Drive** |
| Rainy | Plenty | Heavy | Wet roads, no rush, congested | **Bus** |
| Stormy | Rushed | Heavy | Dangerous conditions | **Work from Home** |
| Cloudy | Plenty | Light | Good weather, relaxed schedule | **Walk** |

---

## 🛠️ Tools & Technologies

- **Machine Learning:** scikit-learn DecisionTreeClassifier
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib with custom styling
- **Environment:** Jupyter Notebook

---

## 📁 Files Structure

```
submission/module-14/14.3/
├── 14.3-submission.ipynb       # Main Jupyter notebook
├── README.md                   # This documentation
└── images/                     # Generated visualizations
    └── work_commute_tree.png   # Decision tree visualization
```

---

## 🚀 How to Run

### 1. Prerequisites:
```bash
pip install pandas numpy scikit-learn matplotlib jupyter
```

### 2. Execute Notebook:
```bash
jupyter notebook 14.3-submission.ipynb
```

### 3. Run All Cells:
- Execute cells sequentially with `Shift + Enter`
- View the decision tree visualization
- Test prediction examples

---

## 📈 Key Insights

### Decision Patterns:
- **Weather is King:** Safety always trumps convenience
- **Time Pressure Matters:** Rushed schedules favor reliable transport (driving)
- **Traffic Awareness:** Heavy congestion shifts preference to public transit
- **Health Conscious:** Good weather + time encourages active transport

### Real-World Validation:
- The model achieves 100% accuracy on the training data
- Decision logic reflects common-sense commute reasoning
- Feature importance aligns with intuitive decision priorities

---

## 🎯 Practical Applications

- **Smart Commute Apps:** Automated daily transport recommendations
- **Corporate Planning:** Understanding employee commute patterns
- **Urban Planning:** Traffic flow and public transit optimization
- **Personal Productivity:** Reducing decision fatigue in morning routines

---

## 🔮 Future Enhancements

- [ ] Real-time weather API integration
- [ ] Live traffic data incorporation
- [ ] Historical commute time analysis
- [ ] Cost factor consideration (gas, parking, transit fares)
- [ ] Environmental impact scoring
- [ ] Seasonal pattern recognition
- [ ] Multi-modal journey planning

---

## Requirements

```
python>=3.8
pandas>=1.3
numpy>=1.21
scikit-learn>=1.0
matplotlib>=3.5
jupyter>=1.0
```

---

## How to Use the Prediction Function

```python
# Example usage
result = predict_commute(
    weather=2,          # Cloudy
    time_available=2,   # Plenty of time
    traffic=1           # Light traffic
)
print(f"Recommended transport: {result}")
# Output: Walk
```

---

**Author:** Gaurav Goel  
**Course:** Berkeley Data Science Program  
**Assignment:** 14.3 - Personal Decision Trees  
**Date:** October 2025

---

*This decision tree demonstrates how everyday decisions can be systematically modeled and validated using machine learning, providing insights into our unconscious decision-making patterns.*