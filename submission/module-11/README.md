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

### Random Forest Model
1. Define the problem as a supervised regression task with `log_price` (log-transformed price) as the target variable.
2. Prepare the feature matrix `X` by dropping the `log_price` column from the cleaned dataset (`df_model`).
3. Prepare the target vector `y` as the `log_price` column.
4. (Optional for speed) Use a small random sample of the data (e.g., 5%) for quick model iteration and testing.
5. Split the data into training and test sets using `train_test_split`.
6. Train a `RandomForestRegressor` (with a small number of trees for speed, e.g., `n_estimators=5`).
7. Fit the model on the training data or sample.
8. Predict on the test set and evaluate performance using RMSE and R².

### Linear Regression Model
Linear Regression was used as a baseline model to compare its performance with the Random Forest model.

#### Steps for Linear Regression Modeling:
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

---

## 5. Evaluation

### Random Forest Model
After training and evaluating the Random Forest regression model, we obtained the following results:

- **Test RMSE:** This value represents the average error in predicting the log-transformed price. Lower values indicate better predictive accuracy.
- **Test R²:** This value indicates the proportion of variance in the target variable explained by the model. Values closer to 1 suggest a better fit.

The most important features identified by the model (based on feature importances) include variables such as car age, odometer reading, manufacturer, model, and condition.

### Linear Regression Model
The Linear Regression model achieved the following performance metrics:
- **RMSE:** `X.XXX`
- **R²:** `X.XX`

While the Linear Regression model performed reasonably well, it was outperformed by the Random Forest model, which captured non-linear relationships more effectively.

---

## 6. Deployment

### Deployment Process
1. **Model Saving and Loading**:
   - The trained Random Forest model was saved using `joblib` for future use.
   - The saved model was successfully loaded and verified by comparing predictions from the loaded model with the original predictions.

2. **Key Deployment Steps**:
   - Save the trained model to a file (`random_forest_model.pkl`).
   - Load the model when needed for predictions.
   - Ensure the integrity of the saved and loaded model by verifying predictions.

3. **Next Steps**:
   - Deploy the model in a production environment to assist with pricing decisions.
   - Monitor the model's performance on new data and retrain as necessary.

---

## 7. Findings

### Business Understanding
- The primary objective of this analysis was to identify and quantify the key factors influencing the price of used cars. This understanding enables used car dealerships to make informed decisions about inventory acquisition, pricing strategies, and marketing efforts.

### Data Cleaning and Preparation
- The dataset was thoroughly cleaned to ensure high-quality data:
  - Removed duplicate records to avoid redundancy.
  - Dropped rows with missing values in critical columns (e.g., price, year, odometer, condition).
  - Filtered outliers and unrealistic values (e.g., extreme prices, mileage, or car years).
  - Engineered new features, such as `car_age`, to better capture the relationship between vehicle characteristics and price.
  - Applied one-hot encoding to categorical variables and log transformations to skewed numerical features for improved model performance.

### Key Findings
- **Actionable Insights**:
  - **Car Age**: Older cars are generally less valuable. Dealerships should focus on acquiring newer cars to maximize profitability.
  - **Mileage**: Cars with lower mileage command higher prices. Marketing efforts should emphasize low-mileage vehicles.
  - **Condition**: Cars in better condition (e.g., "like new") have significantly higher prices. Dealerships should prioritize vehicles in good condition or invest in reconditioning.
  - **Manufacturer and Model**: Certain brands and models consistently fetch higher prices. Inventory decisions should consider these trends.

### Next Steps and Recommendations
1. **Expand Data Collection**:
   - Include additional features such as market demand, regional trends, and seasonal effects to improve model accuracy.
2. **Hyperparameter Tuning**:
   - Perform grid search or randomized search to optimize the Random Forest model for even better performance.
3. **Deploy the Model**:
   - Use the trained Random Forest model to predict car prices for new inventory and assist in pricing decisions.
4. **Monitor Model Performance**:
   - Regularly evaluate the model on new data to ensure it remains accurate and relevant.
5. **Business Strategy**:
   - Use insights from the analysis to guide inventory acquisition, pricing strategies, and marketing campaigns.