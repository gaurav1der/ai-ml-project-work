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

1. Define the problem as a supervised regression task with `log_price` (log-transformed price) as the target variable.
2. Prepare the feature matrix `X` by dropping the `log_price` column from the cleaned dataset (`df_model`).
3. Prepare the target vector `y` as the `log_price` column.
4. (Optional for speed) Use a small random sample of the data (e.g., 5%) for quick model iteration and testing.
5. Split the data into training and test sets using `train_test_split`.
6. Train a `RandomForestRegressor` (with a small number of trees for speed, e.g., `n_estimators=5`).
7. Fit the model on the training data or sample.
8. Predict on the test set and evaluate performance using RMSE and R².

---

## 5. Evaluation

After training and evaluating the Random Forest regression model, we obtained the following results:

- **Test RMSE:** This value represents the average error in predicting the log-transformed price. Lower values indicate better predictive accuracy.
- **Test R²:** This value indicates the proportion of variance in the target variable explained by the model. Values closer to 1 suggest a better fit.

The most important features identified by the model (based on feature importances) include variables such as car age, odometer reading, manufacturer, model, and condition. These features have the greatest influence on predicting used car prices.

**Business Insight:**  
The model suggests that newer cars with lower mileage and in better condition tend to have higher prices. Manufacturer and model also play significant roles, indicating that brand reputation and specific models are valued by consumers.

**Next Steps:**  
1. If model performance is not satisfactory (low R² or high RMSE), consider additional feature engineering, hyperparameter tuning, or trying different algorithms.
2. If the model performs well, these insights can be used to guide inventory decisions and pricing strategies for the dealership.

Overall, the analysis provides actionable information on what drives used car prices, aligning with the business objective.

---

## 6. Deployment

Now that we've settled on our models and findings, it is time to deliver the information to the client. The results should be organized as a clear, concise report that highlights the most important insights:

1. **Summary of Findings:**  
   Present the model's performance metrics (RMSE, R²) and explain what they mean in business terms (e.g., how accurately the model predicts car prices).

2. **Key Drivers of Price:**  
   List the top features influencing used car prices, as identified by the model (such as car age, odometer, condition, manufacturer, and model).

3. **Actionable Recommendations:**  
   Provide guidance for inventory and pricing strategies based on the analysis (e.g., prioritize newer cars with lower mileage and good condition, focus on popular manufacturers/models).

4. **Model Limitations and Next Steps:**  
   Briefly mention any limitations (such as data quality or model assumptions) and suggest future improvements (like collecting more data, trying other algorithms, or refining features).

The report should be tailored for used car dealers, focusing on practical insights that can help them optimize their inventory and pricing decisions.
