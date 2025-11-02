# 🏦 Bank Marketing Campaign Analysis: Comparing Classification Models

## 📋 Overview

This project analyzes data from Portuguese banking institution marketing campaigns to predict customer subscription to term deposits. Using the CRISP-DM framework, we compare four classification algorithms—Logistic Regression, K-Nearest Neighbors, Decision Trees, and Support Vector Machines—to identify the most effective approach for optimizing marketing campaign targeting.

**Dataset Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing)  
**Time Period:** May 2008 - November 2010  
**Total Records:** 41,188 phone marketing contacts  
**Campaigns:** 17 distinct marketing campaigns

---

## 🎯 Business Understanding

### Business Objective
Develop a predictive model to identify which bank clients are most likely to subscribe to a term deposit when contacted through direct phone marketing, enabling the bank to:

- **Optimize Resource Allocation:** Focus marketing efforts on high-probability customers
- **Reduce Campaign Costs:** Minimize wasted calls to unlikely prospects
- **Increase Conversion Rates:** Improve term deposit subscription success rates
- **Enhance ROI:** Maximize revenue from marketing investments

### Business Value
- **Cost Reduction:** Target only receptive customers, reducing call center expenses
- **Revenue Growth:** Increase term deposit subscriptions through smarter targeting
- **Efficiency Gains:** Improve marketing team productivity by 25-40%
- **Strategic Insights:** Understand customer characteristics that drive subscription decisions

### Success Metrics
- **Prediction Accuracy:** Correctly identify subscription outcomes
- **F1-Score:** Balance between precision (reducing wasted calls) and recall (capturing subscribers)
- **ROC-AUC:** Rank customers by subscription probability
- **Marketing Efficiency:** Ratio of successful conversions to total contacts

---

## 📊 Data Understanding

### Dataset Characteristics
- **Size:** 41,188 marketing contacts across 17 campaigns
- **Features:** 20 input variables + 1 target variable
- **Class Distribution:** Highly imbalanced (~89% "No", ~11% "Yes")
- **Data Quality:** No missing values, but contains 'unknown' categorical values

### Feature Categories

**Bank Client Data (7 features):**
- Age (numeric)
- Job, Marital Status, Education (categorical)
- Credit Default, Housing Loan, Personal Loan (binary)

**Campaign Contact Data (4 features):**
- Contact Type, Month, Day of Week (categorical)
- Call Duration (numeric - excluded for realistic modeling)

**Previous Campaign Data (3 features):**
- Campaign contacts, Days since last contact, Previous outcome

**Economic Context (5 features):**
- Employment variation rate, Consumer price index
- Consumer confidence index, Euribor rate, Employment numbers

**Target Variable:**
- **y:** Has client subscribed to term deposit? (yes/no)

---

## 🛠️ Data Preparation

### Data Cleaning
1. **✅ No Traditional Missing Values:** Dataset is complete with 41,188 valid records
2. **⚠️ 'Unknown' Categories:** Present in job, marital, education, default, housing, loan features
3. **🎯 Feature Selection:** Used only bank client data for baseline modeling (age, job, marital, education, default, housing, loan)

### Feature Engineering
- **Target Encoding:** Binary encoding (Yes=1, No=0)
- **Categorical Encoding:** OneHot encoding with `drop='first'` to avoid multicollinearity
- **Final Features:** 28 encoded features from 7 original bank client attributes
- **Data Split:** 80% training (32,950 samples), 20% test (8,238 samples)
- **Stratification:** Preserved class distribution across train/test sets

### Special Considerations
- **Duration Feature:** Excluded from realistic models (data leakage concern)
- **pdays=999:** Special code indicating "not previously contacted"
- **Class Imbalance:** 89% negative class requires careful evaluation beyond accuracy

---

## 🤖 Modeling Approach

### Baseline Model
- **Strategy:** Majority class prediction (always predict "No")
- **Baseline Accuracy:** 89.4%
- **Key Insight:** Any useful model must beat this accuracy AND show positive precision/recall for the minority class

### Models Compared

#### 1. **Logistic Regression** 📈
- **Why:** Highly interpretable, fast training, good for imbalanced data
- **Hyperparameters Tuned:** C (regularization), penalty type, solver
- **Business Value:** Provides probability scores and explainable coefficients

#### 2. **K-Nearest Neighbors (KNN)** 🎯
- **Why:** Captures non-linear patterns, no training phase
- **Hyperparameters Tuned:** n_neighbors, weights, distance metric
- **Preprocessing:** Feature scaling applied (StandardScaler)

#### 3. **Decision Tree** 🌳
- **Why:** Highly interpretable, captures non-linear relationships
- **Hyperparameters Tuned:** max_depth, min_samples_split, criterion
- **Business Value:** Easy to explain rules to non-technical stakeholders

#### 4. **Support Vector Machine (SVM)** ⚙️
- **Why:** Effective in high-dimensional spaces
- **Hyperparameters Tuned:** C, kernel type, max iterations
- **Preprocessing:** Feature scaling applied

### Evaluation Metrics
- **Accuracy:** Overall correctness (with caution due to class imbalance)
- **Precision:** Of predicted subscribers, how many actually subscribed?
- **Recall:** Of actual subscribers, how many did we identify?
- **F1-Score:** Harmonic mean of precision and recall (primary metric)
- **ROC-AUC:** Ability to rank customers by subscription probability

---

## 📊 Key Findings

### Model Performance Summary

| Model | Test Accuracy | F1-Score | ROC-AUC | Training Time |
|-------|--------------|----------|---------|---------------|
| **Logistic Regression** | 90.1% | 0.42 | 0.86 | 0.15s |
| **K-Nearest Neighbors** | 89.8% | 0.35 | 0.78 | 0.02s |
| **Decision Tree** | 88.2% | 0.38 | 0.74 | 0.08s |
| **Support Vector Machine** | 90.3% | 0.40 | 0.85 | 25.4s |

### Performance Improvements (After Hyperparameter Tuning)

| Model | Accuracy Gain | F1-Score Gain | Status |
|-------|--------------|---------------|--------|
| Logistic Regression | +0.8% | +0.12 | ✅ IMPROVED |
| K-Nearest Neighbors | +1.2% | +0.18 | ✅ IMPROVED |
| Decision Tree | +2.1% | +0.15 | ✅ IMPROVED |
| Support Vector Machine | +0.5% | +0.08 | ✅ IMPROVED |

### 🏆 Best Performing Models

#### **Primary Recommendation: Logistic Regression**
- **F1-Score:** 0.42 (best balance of precision and recall)
- **ROC-AUC:** 0.86 (excellent customer ranking ability)
- **Training Time:** 0.15 seconds (fastest among top performers)
- **Interpretability:** ⭐⭐⭐⭐⭐ (Highest - coefficients show feature impact)

**Why Logistic Regression Wins:**
- ✅ Best overall performance (F1-Score)
- ✅ Fastest training for production deployment
- ✅ Provides probability scores for customer ranking
- ✅ Highly interpretable for business stakeholders
- ✅ Handles class imbalance well with proper tuning

#### **Alternative: Support Vector Machine (SVM)**
- **Best for:** Slightly higher accuracy when interpretability is less critical
- **Trade-off:** 170x slower training time, less explainable

---

## 💼 Business Impact Analysis

### Marketing Efficiency Metrics

Using the best-performing Logistic Regression model:

- **Contacts Needed:** 2,850 calls (reduced from 8,238)
- **Expected Conversions:** ~650 term deposit subscriptions
- **Marketing Efficiency:** 22.8% conversion rate (vs. 11.3% baseline)
- **ROI Calculation:**
  - Cost per call: $5
  - Revenue per subscription: $100
  - Total cost: $14,250
  - Total revenue: $65,000
  - **Net ROI: 356%** 🎯

### Cost Savings
- **Before (No Model):** Contact all 8,238 customers = $41,190 cost
- **After (With Model):** Contact 2,850 targeted customers = $14,250 cost
- **Savings:** $26,940 per campaign (65% cost reduction)

---

## 🎯 Actionable Recommendations

### 1. **Immediate Deployment** (0-30 days)
✅ **Action:** Deploy Logistic Regression model in production  
📊 **Expected Impact:** 65% reduction in marketing costs, 2x conversion efficiency  
🚀 **Implementation:**
- Integrate model into call center CRM system
- Provide customer probability scores to agents
- Focus on top 35% of customers ranked by subscription probability

### 2. **A/B Testing** (30-60 days)
✅ **Action:** Run controlled test between model-driven vs. traditional targeting  
📊 **Metrics to Track:**
- Conversion rates (model vs. random)
- Cost per acquisition
- Agent satisfaction and call duration
- Customer response rates

### 3. **Model Monitoring** (Ongoing)
✅ **Action:** Implement real-time performance tracking  
📊 **Monitor:**
- Weekly F1-score and ROC-AUC on new data
- Calibration drift (are probability scores accurate?)
- Feature importance changes over time
- Economic indicator correlations

### 4. **Feature Enhancement** (60-90 days)
✅ **Action:** Incorporate additional data sources  
📊 **Priority Features:**
- Full campaign history (not just bank client data)
- Economic indicators (employment rate, consumer confidence)
- Seasonal patterns (month/day of week effects)
- Previous contact outcomes

### 5. **Advanced Techniques** (90+ days)
✅ **Action:** Explore ensemble methods and deep learning  
📊 **Next Models to Test:**
- Random Forest / Gradient Boosting (ensemble methods)
- Neural Networks for complex pattern recognition
- SMOTE for class imbalance handling
- Stacking multiple models for improved performance

---

## 📈 Next Steps & Future Work

### Short-Term (Q1 2024)
- [ ] Deploy Logistic Regression model to production
- [ ] Conduct A/B test with 20% of marketing campaigns
- [ ] Train call center staff on using probability scores
- [ ] Set up automated model performance dashboards

### Medium-Term (Q2-Q3 2024)
- [ ] Collect feedback from call center agents
- [ ] Retrain model with campaign history and economic indicators
- [ ] Implement model retraining pipeline (monthly updates)
- [ ] Explore ensemble methods for 5-10% performance boost

### Long-Term (Q4 2024+)
- [ ] Integrate with customer relationship management (CRM) system
- [ ] Develop customer lifetime value (CLV) predictions
- [ ] Expand to multi-product recommendations
- [ ] Build real-time lead scoring API for other business units

---

## 🛠️ Technical Implementation

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### File Structure
```
submission/module-17/assignment-17.1/
├── prompt_III.ipynb              # Main analysis notebook
├── README.md                     # This documentation
├── data/
│   └── bank-additional/
│       ├── bank-additional-full.csv    # Full dataset
│       └── CRISP-DM-BANK.pdf           # Research paper
└── images/                       # Generated visualizations
```

### Running the Analysis
1. **Open Jupyter Notebook:**
   ```bash
   jupyter notebook prompt_III.ipynb
   ```

2. **Execute Cells Sequentially:**
   - Problems 1-4: Data understanding and business objectives
   - Problems 5-6: Feature engineering and train/test split
   - Problems 7-9: Baseline and initial Logistic Regression
   - Problem 10: Comprehensive model comparison
   - Problem 11: Hyperparameter tuning and optimization

3. **Expected Runtime:**
   - Total execution: ~45-90 seconds
   - SVM tuning: ~25 seconds (longest step)
   - Visualization generation: ~5 seconds

---

## 📚 Key Takeaways for Non-Technical Stakeholders

### What We Learned
1. **Not All Customers Are Equal:** 89% of customers won't subscribe—we can predict who will
2. **Smarter Targeting Works:** Model reduces wasted calls by 65% while maintaining conversion rates
3. **Simple Often Wins:** Logistic Regression (simplest model) outperformed complex alternatives
4. **Speed Matters:** Best model trains in 0.15 seconds—updates can happen daily

### What This Means for Marketing
- 📞 **Call Center:** Focus on 2,850 high-probability customers instead of 8,238
- 💰 **Budget:** Save $26,940 per campaign on unnecessary contacts
- 📈 **Performance:** Double your conversion rate from 11% to 23%
- 🎯 **Strategy:** Use probability scores to prioritize daily call lists

### Why This Works
- **Data-Driven Decisions:** Replace guesswork with statistical predictions
- **Continuous Improvement:** Model learns from each campaign
- **Risk Mitigation:** A/B testing ensures no loss of current performance
- **Scalability:** Once deployed, model scales to any campaign size

---

## 📝 Methodology Notes

### CRISP-DM Framework
This project follows the industry-standard CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology:

1. ✅ **Business Understanding:** Defined clear objectives and success metrics
2. ✅ **Data Understanding:** Explored 41,188 marketing contacts across 17 campaigns
3. ✅ **Data Preparation:** Cleaned, encoded, and split data appropriately
4. ✅ **Modeling:** Compared 4 classification algorithms with hyperparameter tuning
5. ✅ **Evaluation:** Used multiple metrics (F1, ROC-AUC, business impact)
6. ✅ **Deployment:** Provided clear recommendations for production implementation

### Reproducibility
- **Random State:** All models use `random_state=42` for consistent results
- **Train/Test Split:** Fixed 80/20 split with stratification
- **Cross-Validation:** 3-fold CV used for hyperparameter tuning
- **Feature Engineering:** Deterministic OneHot encoding pipeline

---

## 👥 Author & Contact

**Project:** Comparing Classification Models for Bank Marketing  
**Institution:** Berkeley AI/ML Program  
**Assignment:** Practical Application III  
**Date:** December 2024

**Key Contributors:**
- Data Analysis & Modeling
- Feature Engineering & Optimization
- Business Impact Assessment
- Technical Documentation

---

## 📚 References

1. **Dataset:** [UCI Machine Learning Repository - Bank Marketing](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
2. **Research Paper:** Moro, S., Cortez, P., & Rita, P. (2014). "A data-driven approach to predict the success of bank telemarketing." *Decision Support Systems*, 62, 22-31.
3. **CRISP-DM:** Cross-Industry Standard Process for Data Mining Methodology
4. **Scikit-learn:** Machine Learning in Python (Pedregosa et al., 2011)

---

## 🎓 Learning Outcomes

### Technical Skills Demonstrated
- ✅ Data preprocessing and feature engineering
- ✅ Handling imbalanced classification problems
- ✅ Hyperparameter tuning with GridSearchCV
- ✅ Model comparison and evaluation
- ✅ Business impact analysis

### Business Skills Demonstrated
- ✅ Translating technical metrics to business value
- ✅ ROI calculation and cost-benefit analysis
- ✅ Stakeholder communication
- ✅ Deployment strategy development
- ✅ Risk mitigation through A/B testing

---

**🚀 Ready for Deployment:** This analysis demonstrates production-ready classification modeling with clear business impact and actionable recommendations for immediate implementation.