# 🔍 Grid Search Methods Comparison for Decision Trees

## 📋 Overview

This notebook compares four different hyperparameter optimization techniques for DecisionTreeClassifier using the Whickham dataset to understand their trade-offs between speed, thoroughness, and performance. The analysis demonstrates how to choose the right grid search method based on your computational constraints and dataset characteristics.

---

## 🎯 Objective

Compare the effectiveness and efficiency of:
- **GridSearchCV** - Exhaustive search of all parameter combinations
- **RandomizedSearchCV** - Random sampling approach
- **HalvingGridSearchCV** - Iterative elimination with full grid
- **HalvingRandomSearchCV** - Random sampling with early elimination

---

## 📊 Dataset: Whickham Survival Study

### Dataset Overview
- **Source:** Whickham.txt - A longitudinal study of smoking and survival
- **Features:** 
  - `age` - Age of participant (numerical)
  - `smoker_encoded` - Smoking status (0=No, 1=Yes)
- **Target:** `outcome` - Survival status (0=Alive, 1=Dead)
- **Size:** 1,314 total samples
- **Train/Test Split:** 80/20 with random_state=42

### Real-world Relevance
This dataset provides a meaningful binary classification problem predicting survival outcomes based on age and smoking status, making it ideal for demonstrating decision tree hyperparameter optimization in a medical context.

---

## 🔧 Technical Setup

### Parameter Grid
```python
params = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 10, 20],
    'criterion': ['gini', 'entropy']
}
# Total combinations: 24
```

### Evaluation Metrics
- **⏱️ Execution Time** - Time to complete hyperparameter search
- **🎯 Best Score** - Highest cross-validation accuracy achieved
- **📊 Total Fits** - Number of models trained during search
- **⚡ Efficiency** - Score per second ratio (overall value)

---

## 🚀 Grid Search Methods Comparison

### 1️⃣ GridSearchCV
- **Strategy:** Tests ALL parameter combinations (24 total)
- **Pros:** Most thorough, guaranteed to find global optimum
- **Cons:** Computationally expensive, doesn't scale well
- **Best For:** Small parameter spaces, critical medical applications

### 2️⃣ RandomizedSearchCV
- **Strategy:** Random sampling of parameter combinations (15 iterations)
- **Pros:** Good balance of speed and quality, scales well
- **Cons:** May miss optimal combinations, requires setting n_iter
- **Best For:** Large parameter spaces, exploratory analysis

### 3️⃣ HalvingGridSearchCV
- **Strategy:** Iterative elimination of poor performers
- **Pros:** Systematic approach, faster than full grid
- **Cons:** May eliminate good parameters early, experimental feature
- **Best For:** Medium datasets, systematic optimization

### 4️⃣ HalvingRandomSearchCV
- **Strategy:** Random sampling + early elimination
- **Pros:** Fastest method, good for large datasets
- **Cons:** Most aggressive elimination, experimental feature
- **Best For:** Large datasets, maximum efficiency needed

---

## 📊 Visualization & Analysis

The notebook generates comprehensive comparisons:

### Four-Panel Analysis
1. **⏱️ Execution Time Comparison** - Bar chart of search times
2. **🎯 Best Score Comparison** - Accuracy achievements
3. **📊 Total Model Fits** - Computational cost visualization
4. **⚡ Efficiency Comparison** - Score per second ratios

### Detailed Results Table
- Time, Score, Total Fits, and Efficiency for each method
- Clear identification of fastest, highest scoring, and most efficient methods

---

## 💡 Expected Results & Insights

### 🎯 Performance Patterns
- **All methods** typically achieve similar accuracy scores (within 1-2%)
- **GridSearchCV** tests all 24 combinations systematically
- **RandomizedSearchCV** tests 15 random combinations
- **Halving methods** use fewer total fits through elimination
- **Time differences** can be 2-5x between fastest and slowest methods

### 🔍 Decision Framework

| Scenario | Recommended Method | Reasoning |
|----------|-------------------|-----------|
| **Small Dataset** (< 1K samples) | GridSearchCV | Thoroughness matters more than speed |
| **Medium Dataset** (1K-10K) | RandomizedSearchCV | Balanced approach |
| **Large Dataset** (> 10K) | HalvingRandomSearchCV | Efficiency critical |
| **Medical/Critical Apps** | GridSearchCV | Need guaranteed optimal results |
| **Exploration Phase** | RandomizedSearchCV | Good coverage of parameter space |
| **Production Environment** | HalvingRandomSearchCV | Time constraints |

### ⚙️ Key Insights
- **Feature Importance:** Age typically dominates smoking status in survival prediction
- **Optimal Parameters:** Often include moderate max_depth (5-10) and gini criterion
- **Efficiency Trade-offs:** Halving methods provide 80% of the performance in 40% of the time
- **Parallel Processing:** `n_jobs=-1` significantly improves all methods

---

## 🛠️ Tools & Libraries

```python
# Core Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Sklearn Components
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     HalvingGridSearchCV, HalvingRandomSearchCV)
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_halving_search_cv
```

---

## 📁 File Structure

```
submission/module-14/14.1/
├── 14.1-submission.ipynb    # Main analysis notebook
├── README.md                # This documentation
├── data/                    # Dataset files
│   └── Whickham.txt         # Survival study data
└── images/                  # Generated visualizations
    └── search_comparison.jpg # Four-panel comparison chart
```

---

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas numpy matplotlib scikit-learn
```

### Execution
1. Open the Jupyter notebook: `14.1-submission.ipynb`
2. Ensure `data/Whickham.txt` is in the correct path
3. Run all cells sequentially
4. View comparative visualizations and analysis
5. Review recommendations in the final summary

### Expected Runtime
- **Total execution time:** 15-45 seconds (depends on hardware)
- **GridSearchCV:** Longest (exhaustive search)
- **HalvingRandomSearchCV:** Shortest (early elimination)

---

## 📈 Expected Results

### Typical Performance Patterns
- **GridSearchCV:** ~0.74-0.76 accuracy, longest time (24 fits)
- **RandomizedSearchCV:** ~0.73-0.75 accuracy, moderate time (15 fits)
- **HalvingGridSearchCV:** ~0.73-0.75 accuracy, faster than grid (12-18 fits)
- **HalvingRandomSearchCV:** ~0.72-0.74 accuracy, shortest time (8-12 fits)

### Success Metrics
- All methods typically achieve similar accuracy (within 2%)
- Time differences show clear efficiency advantages of halving methods
- Efficiency ratios demonstrate best value propositions for different scenarios

---

## 🎯 Learning Outcomes

### Technical Skills
- **Hyperparameter Optimization:** Understanding different search strategies
- **Performance Analysis:** Comparing time vs. accuracy trade-offs
- **Medical Data Processing:** Working with survival/mortality datasets
- **Visualization:** Creating comprehensive comparison charts
- **Method Selection:** Choosing appropriate techniques for different scenarios

### Practical Applications
- **Medical Research:** Efficient model optimization for clinical decision trees
- **Production Deployment:** Time-constrained optimization strategies
- **Resource Management:** Balancing computational cost with model quality
- **Research Projects:** Systematic comparison of optimization methods

---

## 🔮 Future Extensions

- [ ] **Cross-validation Strategy:** Compare different CV folds (3, 5, 10)
- [ ] **Larger Parameter Grids:** Test with more complex parameter spaces
- [ ] **Different Algorithms:** Compare across Random Forest, SVM, etc.
- [ ] **Bayesian Optimization:** Include modern optimization techniques
- [ ] **Multi-objective Optimization:** Balance accuracy, interpretability, and speed
- [ ] **Real-time Analysis:** Implement online hyperparameter optimization

---

## 📊 Key Takeaways

### Method Selection Guide
1. **Time Budget** - How long can you wait for results?
2. **Dataset Size** - Larger datasets favor halving methods
3. **Parameter Space Size** - Larger spaces favor random methods
4. **Application Criticality** - Medical/financial applications favor exhaustive search
5. **Available Resources** - Computational constraints guide method choice

### Performance Summary
- **Best Accuracy:** Usually GridSearchCV (most thorough)
- **Best Speed:** Usually HalvingRandomSearchCV (smart elimination)
- **Best Balance:** Usually RandomizedSearchCV (good compromise)
- **Best for Production:** HalvingRandomSearchCV (efficiency + good results)

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

**Author:** Gaurav Goel  
**Course:** Berkeley Data Science Program  
**Assignment:** 14.1 - Grid Searching Decision Trees  
**Dataset:** Whickham Survival Study  
**Date:** October 2025

---

*This analysis provides practical guidance for choosing hyperparameter optimization methods based on computational constraints and dataset characteristics, demonstrated through a real-world medical dataset predicting survival outcomes.*