# Used Car Price Prediction: CRISP-DM Approach

## 1. Business Understanding

The business goal is to identify the key factors that influence used car prices, enabling a dealership to make data-driven decisions about pricing, inventory, and marketing.

---

## 2. Data Understanding

- Load and inspect `vehicles.csv`.
- Explore the dataset’s features (e.g., year, make, model, mileage, condition, transmission, fuel type, etc.).
- Visualize distributions and relationships, especially with respect to the `price` column.
- Identify missing values, outliers, and data quality issues.

---

## 3. Data Preparation

1. **Remove Duplicates:**  
   Drop duplicate rows to ensure each record is unique.

2. **Drop Rows with Missing Target or Critical Features:**  
   Remove rows with missing values in `price`, `year`, `odometer`, or `condition`.

3. **Filter Outliers and Unrealistic Values:**  
   - Keep only records with `price` between \$500 and \$100,000.
   - Keep only records with `odometer` between 0 and 500,000.
   - Keep only records with `year` between 1980 and 2025.

4. **Feature Engineering:**  
   - Create a new feature `car_age` as `current_year - year`.

5. **Encode Categorical Variables:**  
   - Use one-hot encoding (`pd.get_dummies`) for categorical columns such as `year`, `model`, `condition`, `fuel`, `transmission`, `type`, `drive`, and `paint_color`.
   - For additional modeling, also encode columns like `cylinders`, `region`, `manufacturer`, `make`, `title_status`, `size`, and `state` if present.

6. **Transform Features:**  
   - Apply log transformation to `odometer` and `price` to reduce skewness (`log_odometer`, `log_price`).

7. **Drop Unnecessary Columns:**  
   Remove columns not needed for modeling, such as `id`, `url`, `region_url`, `VIN`, `county`, `lat`, `long`, `image_url`, `description`, `year`, `odometer`, and `price`.

8. **Final Dataset Construction:**  
   The cleaned and transformed dataset (`df_model`) is now ready for modeling, with all features numeric and relevant for regression.

---

## 4. Modeling

### 4.1 Random Forest Model
1. Define the problem as a supervised regression task with `log_price` (log-transformed price) as the target variable.
2. Prepare the feature matrix `X` by dropping the `log_price` column from the cleaned dataset (`df_model`).
3. Prepare the target vector `y` as the `log_price` column.
4. (Optional for speed) Use a small random sample of the data (e.g., 5%) for quick model iteration and testing.
5. Split the data into training and test sets using `train_test_split`.
6. Train a `RandomForestRegressor` (with a small number of trees for speed, e.g., `n_estimators=5`).
7. Fit the model on the training data or sample.
8. Predict on the test set and evaluate performance using RMSE and R².

### 4.2 Linear Regression Model
Linear Regression was used as a baseline model to compare its performance with the Random Forest model.

**Steps for Linear Regression Modeling:**
1. **Data Sampling**:
   - To speed up training, 1% of the dataset was used for training and testing.
   - The data was split into training and testing sets using `train_test_split`.

2. **Model Training**:
   - A `LinearRegression` model from `sklearn` was instantiated and trained on the sampled training data.

3. **Predictions**:
   - Predictions were made on a 10% sample of the test dataset to evaluate the model's performance.

4. **Evaluation**:
   - The model's performance was evaluated using the following metrics:
     - **RMSE (Root Mean Squared Error)**: Measures the average prediction error in the same units as the target variable.
     - **R² (R-squared)**: Indicates how well the model explains the variance in the target variable.

5. **Cross-Validation**:
   - A 3-fold cross-validation was performed to validate the model's performance across different subsets of the data.

### 4.3 HistGradientBoostingRegressor Model
The **HistGradientBoostingRegressor** is a fast and efficient implementation of gradient boosting for regression tasks. It is particularly well-suited for large datasets and provides competitive performance with reduced training time.

**Key Features:**
- **Efficiency**: The histogram-based implementation significantly reduces training time compared to traditional gradient boosting methods.
- **Performance**: Captures non-linear relationships between features and the target variable effectively.
- **Scalability**: Handles large datasets efficiently.

**Model Configuration:**
- **Max Iterations**: 50
- **Max Depth**: 3
- **Random State**: 42 (for reproducibility)

**Evaluation Metrics:**
- **RMSE (Root Mean Squared Error)**: 0.372
- **R² (R-squared)**: 0.862

**Conclusion:**
The **HistGradientBoostingRegressor** is the best-performing model in this project, offering both speed and accuracy. It is well-suited for large datasets and provides a good balance between performance and computational efficiency.

---

## 5. Grid Search for Hyperparameters

This project uses grid search to optimize the hyperparameters of all regression models. Grid search is performed using scikit-learn's `GridSearchCV` or `RandomizedSearchCV`, which systematically tests different combinations of model parameters to find the best settings based on cross-validated performance.

### 5.1 How Grid Search Works

- **Parameter Grid:** Define a dictionary of hyperparameters and their possible values for each model.
- **Cross-Validation:** For each combination, the model is trained and validated using cross-validation.
- **Scoring:** The best combination is selected based on a scoring metric (e.g., negative mean squared error).

### 5.2 Example Usage

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [5, 10, None]
}
grid = GridSearchCV(RandomForestRegressor(), param_grid, cv=3, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)
print("Best parameters:", grid.best_params_)
```

### 5.3 Tips for Large Datasets

- Use a smaller sample of your data for grid search to reduce execution time.
- Limit the number of parameter combinations and cross-validation folds.
- For large grids, use `RandomizedSearchCV` to randomly sample parameter combinations.

See the notebook for detailed grid search examples for Linear Regression, Random Forest, and HistGradientBoostingRegressor.

--- 
## 6. Interpretation of Linear Regression Coefficients, Regression Model Coefficients and Feature Importances

### 6.1 Linear Regression Coefficients

In a linear regression model, each coefficient represents the expected change in the target variable (here, the log-transformed car price) for a one-unit increase in the corresponding feature, holding all other features constant.

- **Positive coefficients**: As the feature increases, the predicted price increases.
- **Negative coefficients**: As the feature increases, the predicted price decreases.
- **Magnitude**: The absolute value of the coefficient indicates the strength of the relationship between the feature and the target variable.

Coefficients are especially useful for understanding which features have the most significant impact on the target and for interpreting the direction and size of these effects.

---

### 6.2 Random Forest and Tree-Based Models: Feature Importances

Tree-based models like RandomForestRegressor do not provide coefficients, but instead offer a `feature_importances_` attribute. This attribute reflects how much each feature contributes to reducing prediction error across all trees in the ensemble.

- **Higher importance**: The feature plays a larger role in the model’s predictions.
- **Lower importance**: The feature has less influence on the outcome.

Feature importances help identify which variables are most influential in the model, but do not indicate the direction (positive or negative) of the relationship.

---

### 6.3 HistGradientBoostingRegressor: Permutation Importance

The `HistGradientBoostingRegressor` does not provide a `feature_importances_` attribute. Instead, feature importance can be assessed using **permutation importance**:

- **Permutation importance** measures the decrease in model performance when the values of a single feature are randomly shuffled.
- A larger decrease in performance indicates a more important feature.

Permutation importance provides a model-agnostic way to interpret which features are most influential, even for models that do not natively support feature importances.

---

**Summary Table**

| Model Type                | How to Interpret Importance         | Directionality | Output Attribute/Method         |
|---------------------------|-------------------------------------|----------------|---------------------------------|
| Linear Regression         | Coefficient value                   | Yes            | `coef_`                        |
| Random Forest Regressor   | Feature importance (relative)       | No             | `feature_importances_`          |
| HistGradientBoostingRegressor | Permutation importance (relative) | No             | `permutation_importance()`      |

Understanding these outputs helps you interpret which features drive predictions in each model and supports data-driven business decisions.

---

## 7. Evaluation

### 7.1 Model Comparison Summary

This section summarizes the performance of the three regression models used in the analysis: **Linear Regression**, **Random Forest Regressor**, and **HistGradientBoostingRegressor**. The models were evaluated based on two key metrics:
- **RMSE (Root Mean Squared Error)**: Measures the average prediction error in the same units as the target variable. Lower values indicate better performance.
- **R² (R-squared)**: Indicates how well the model explains the variance in the target variable. Values closer to 1 suggest a better fit.

**Results:**

| Model                        | RMSE   | R²    | Key Observations                                                                 |
|------------------------------|--------|-------|----------------------------------------------------------------------------------|
| **Random Forest Regressor**  | 0.460  | 0.725 | - Captures non-linear relationships effectively.                                |
|                              |        |       | - Provides feature importance for interpretability.                             |
|                              |        |       | - Computationally expensive for large datasets.                                 |
| **Linear Regression**        | 0.745  | 0.279 | - Simple and interpretable.                                                     |
|                              |        |       | - Struggles to capture non-linear relationships in the data.                    |
|                              |        |       | - Sensitive to multicollinearity and outliers.                                  |
| **HistGradientBoostingRegressor** | 0.473  | 0.709 | - Fast and efficient for large datasets.                                        |
|                              |        |       | - Captures non-linear relationships well.                                       |
|                              |        |       | - Requires careful hyperparameter tuning for optimal performance.               |

**Key Findings:**

1. **Random Forest Regressor**:
   - Achieved better performance than Linear Regression by capturing non-linear patterns.
   - Computationally more expensive, especially for large datasets.
2. **Linear Regression**:
   - Performed the worst among the three models due to its inability to capture non-linear relationships in the data.
   - Useful as a baseline model for comparison.
3. **HistGradientBoostingRegressor**:
   - Outperformed both Linear Regression and Random Forest in terms of speed and accuracy.
   - Well-suited for large datasets and provides a good balance between performance and computational efficiency.

**Conclusion:**
- The **HistGradientBoostingRegressor** is the best-performing model for this dataset, offering both speed and accuracy.
- While **Random Forest Regressor** is also effective, it is slower and less efficient for large datasets.
- **Linear Regression** serves as a simple baseline but is not suitable for capturing complex relationships in the data.

---

## 8. Deployment

### 8.1 Deployment Process
1. **Model Saving and Loading**:
   - The trained models were saved using `joblib` for future use.
   - The saved models were successfully loaded and verified by comparing predictions from the loaded models with the original predictions.

2. **Key Deployment Steps**:
   - Save the trained models to files (e.g., `random_forest_model.pkl`, `hist_gradient_boosting_model.pkl`).
   - Load the models when needed for predictions.
   - Ensure the integrity of the saved and loaded models by verifying predictions.

3. **Next Steps**:
   - Deploy the best-performing model (**HistGradientBoostingRegressor**) in a production environment to assist with pricing decisions.
   - Monitor the model's performance on new data and retrain as necessary.

---

## 9. Findings

### 9.1 Business Understanding
- The primary objective of this analysis was to identify and quantify the key factors influencing the price of used cars. This understanding enables used car dealerships to make informed decisions about inventory acquisition, pricing strategies, and marketing efforts.

### 9.2 Data Cleaning and Preparation
- The dataset was thoroughly cleaned to ensure high-quality data:
  - Removed duplicate records to avoid redundancy.
  - Dropped rows with missing values in critical columns (e.g., price, year, odometer, condition).
  - Filtered outliers and unrealistic values (e.g., extreme prices, mileage, or car years).
  - Engineered new features, such as `car_age`, to better capture the relationship between vehicle characteristics and price.
  - Applied one-hot encoding to categorical variables and log transformations to skewed numerical features for improved model performance.

### 9.3 Key Findings
- **Actionable Insights**:
  - **Car Age**: Older cars are generally less valuable. Dealerships should focus on acquiring newer cars to maximize profitability.
  - **Mileage**: Cars with lower mileage command higher prices. Marketing efforts should emphasize low-mileage vehicles.
  - **Condition**: Cars in better condition (e.g., "like new") have significantly higher prices. Dealerships should prioritize vehicles in good condition or invest in reconditioning.
  - **Manufacturer and Model**: Certain brands and models consistently fetch higher prices. Inventory decisions should consider these trends.

### 9.4 Next Steps and Recommendations
1. **Expand Data Collection**:
   - Include additional features such as market demand, regional trends, and seasonal effects to improve model accuracy.
2. **Hyperparameter Tuning**:
   - Perform grid search or randomized search to optimize the models for even better performance.
3. **Deploy the Model**:
   - Use the trained **HistGradientBoostingRegressor** model to predict car prices for new inventory and assist in pricing decisions.
4. **Monitor Model Performance**:
   - Regularly evaluate the model on new data to ensure it remains accurate and relevant.
5. **Business Strategy**:
   - Use insights from the analysis to guide inventory acquisition, pricing strategies, and marketing campaigns.

### ## Jupyter notebook

[module11.ipynb](/submission/module-11/module11.ipynb)