# 🏠 House Price Prediction

## 📌 Project Overview
This project predicts house prices using machine learning models. We processed real estate data, handled missing values and outliers, and applied feature engineering to improve prediction accuracy. The best-performing model was **Linear Regression**, achieving an **R² score of 0.678**.

## 📊 Dataset
- **Source:** [UCI Housing Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data)
- **Features:** Area, Bedrooms, Bathrooms, Stories, Parking, etc.
- **Target:** House Price (Log-Transformed)

## ⚡ Models & Performance
| Model               | R² Score | RMSE |
|---------------------|---------|------|
| **Linear Regression**  | **0.678**    | **0.249** |
| Random Forest      | 0.632    | 0.267 |
| XGBoost           | 0.623    | 0.270 |

✅ **Linear Regression performed the best**, meaning house prices in this dataset follow a **mostly linear trend**.

---

## 🎯 Actual vs. Predicted Prices
Below are a few **sample predictions** from the trained model:

| **Actual Price** | **Predicted Price** |
|-----------------|------------------|
| $4,060,000 | $4,826,033 |
| $6,650,000 | $7,425,972 |
| $3,710,000 | $3,333,629 |
| $6,440,000 | $4,556,853 |
| $2,800,000 | $3,407,056 |

📈 **Scatter Plot of Predictions**  
<img src="reports/actual_vs_predicted.png" width="500"/>

✅ **Most predictions are close to actual prices, with minor variance.**

---

## 🔑 Key Insights
### 📌 **Feature Importance (Random Forest)**
| Feature | Importance |
|---------|------------|
| **log_area** | High |
| **bathrooms** | High |
| **unfurnished** | Medium |
| **airconditioning** | Medium |
| **stories** | Medium |
| **bedrooms** | Low |

- **House area (`log_area`) is the most significant factor affecting price.**
- **Bathrooms, furnishing status, and air conditioning also impact pricing.**

---

## 🚀 How to Run
1️⃣ **Clone this repository:**
```sh
git clone https://github.com/alookayparathay/house-price-prediction.git
cd house-price-prediction
