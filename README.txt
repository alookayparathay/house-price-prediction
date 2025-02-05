# 🏠 House Price Prediction

## 📌 Project Overview
This project predicts house prices using Machine Learning models (Linear Regression, Random Forest, XGBoost).

## 📊 Dataset
- **Source:** [UCI Housing Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data)
- **Features:** Area, Bedrooms, Bathrooms, Stories, Parking, etc.
- **Target:** House Price (Log-Transformed)

## ⚡ Models & Performance
| Model               | R² Score | RMSE |
|---------------------|---------|------|
| Linear Regression  | **0.678**    | **0.249** |
| Random Forest      | 0.632    | 0.267 |
| XGBoost           | 0.623    | 0.270 |

✅ **Linear Regression performed the best**, meaning the relationship is mostly **linear**.

## 🔑 Key Findings
- `log_area` and `area` had the **highest feature importance**.
- Bathrooms, Air Conditioning, and Stories **also influenced price**.
- `bedrooms` had **lower impact** than expected.

## 🚀 How to Run
1. Install dependencies:  
   ```sh
   pip install -r requirements.txt
