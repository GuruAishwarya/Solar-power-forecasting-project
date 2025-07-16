#  Forecasting of Solar Power Generation

This project focuses on forecasting solar power generation using machine learning and time series models. The dataset used spans the **entire year of 2006**, recorded at **5-minute intervals**, and contains two columns:
- `timestamp`
- `power`

## 📊 Dataset
- **Source**: [Your dataset source, if public]
- **Structure**: 1 year of data (2006) with power generation values at 5-minute intervals.
- **Preprocessing**: The data was divided into:
  - **Training set**: January to October
  - **Testing/Forecasting set**: November

## 🧠 Models Used
Four models were used for forecasting:
1. **Random Forest Regressor**
2. **LightGBM**
3. **XGBoost**
4. **Facebook Prophet**

## 🧪 Evaluation Metrics
The performance of each model was evaluated using the following metrics:
- **RMSE (Root Mean Squared Error)**
- **MAPE (Mean Absolute Percentage Error)**

## ✅ Best Performing Model
Among all models tested, the **Random Forest Regressor** showed the best performance based on both RMSE and MAPE values.

## 📈 Results Summary
| Model               | RMSE       | MAPE       |
|---------------------|------------|------------|
| Random Forest       |    5.65    |    0.99    |
| LightGBM            |    6.65    |    1.02    |
| XGBoost             |    6.73    |    1.08    |
| Prophet             |    6.62    |    1.28    |


