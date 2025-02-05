import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ✅ Load the dataset
file_path = r"E:\Datasets\house_prices.csv"  
data = pd.read_csv(file_path, encoding="utf-8")

# ✅ Convert 'yes'/'no' to 1/0
yes_no_columns = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
data[yes_no_columns] = data[yes_no_columns].replace({"yes": 1, "no": 0})

# ✅ One-Hot Encoding for Furnishing Status
data = pd.get_dummies(data, columns=["furnishingstatus"], drop_first=True)

# ✅ Convert Boolean Columns to Integers (Ensuring Consistency)
binary_columns = ["furnishingstatus_semi-furnished", "furnishingstatus_unfurnished"]
data[binary_columns] = data[binary_columns].astype(int)

# ✅ Identify Outliers Using IQR
Q1 = data["price"].quantile(0.25)
Q3 = data["price"].quantile(0.75)
IQR = Q3 - Q1
outliers = data[(data["price"] < (Q1 - 1.5 * IQR)) | (data["price"] > (Q3 + 1.5 * IQR))]
print(f"🔍 {len(outliers)} Outliers detected in 'price' column.")

# ✅ Apply Log Transformation (To Reduce Skewness)
data["log_price"] = np.log1p(data["price"])
data["log_area"] = np.log1p(data["area"])

# ✅ Drop Original Price Column (Using log_price Instead)
data.drop(columns=["price"], inplace=True)

# ✅ Boxplot After Log Transformation
plt.figure(figsize=(12, 6))
sns.boxplot(data=data[["log_price", "log_area"]])
plt.title("Boxplot After Log Transformation")
plt.show()

# ✅ Select Features for Scaling
num_features = ["area", "bedrooms", "bathrooms", "stories", "parking"]

# ✅ Apply Standardization
scaler = StandardScaler()
data[num_features] = scaler.fit_transform(data[num_features])

# ✅ Split Data Into Training & Testing Sets
X = data.drop(columns=["log_price"])  # Features
y = data["log_price"]  # Target Variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"✅ Data Preprocessing Complete! Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ----------------- Model Training & Evaluation -----------------

# ✅ Define Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
}

# ✅ Train & Evaluate Models
results = []
for name, model in models.items():
    model.fit(X_train, y_train)  # Train the model
    y_pred = model.predict(X_test)  # Predict on test set
    
    r2 = r2_score(y_test, y_pred)  # R² Score
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # RMSE
    
    results.append([name, r2, rmse])
    print(f"{name} → R²: {r2:.3f}, RMSE: {rmse:.3f}")

# ✅ Convert results to DataFrame for better visualization
results_df = pd.DataFrame(results, columns=["Model", "R² Score", "RMSE"])
print("\n📊 Model Performance Comparison:")
print(results_df)

# ----------------- Generate Predictions on Test Data -----------------

# ✅ Use the best model (Linear Regression) to make predictions
best_model = models["Linear Regression"]
y_pred = best_model.predict(X_test)

# ✅ Convert log-transformed predictions back to original price scale
y_pred_original = np.expm1(y_pred)
y_test_original = np.expm1(y_test)

# ✅ Create a DataFrame to compare actual vs. predicted prices
predictions_df = pd.DataFrame({
    "Actual Price": y_test_original,
    "Predicted Price": y_pred_original
})

# ✅ Display first few predictions
print("\n🎯 Actual vs. Predicted Prices:")
print(predictions_df.head())

# ✅ Scatter Plot: Actual vs. Predicted Prices
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test_original, y=y_pred_original)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs. Predicted House Prices")
plt.show()

# ----------------- Hyperparameter Tuning for Random Forest -----------------

# ✅ Define hyperparameter grid
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10]
}

# ✅ Perform Grid Search
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring="r2")
grid_search.fit(X_train, y_train)

# ✅ Print best parameters and score
print("\n🔧 Hyperparameter Tuning Results:")
print("Best Parameters:", grid_search.best_params_)
print("Best R² Score:", grid_search.best_score_)

# ----------------- Feature Importance (Random Forest) -----------------

# ✅ Get feature importance from Random Forest
importances = models["Random Forest"].feature_importances_

# ✅ Plot feature importance
plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=X.columns)
plt.title("Feature Importance (Random Forest)")
plt.show()

# ----------------- Save the Best Model -----------------

# ✅ Save Linear Regression model
joblib.dump(best_model, "house_price_model.pkl")
print("\n✅ Model saved successfully as 'house_price_model.pkl'!")
