# Student-Performance-Prediction-System
This is a system which will predict the student performance based on data required for the model.
# Student Performance Prediction System

## Project Abstract

This project implements an end-to-end Machine Learning web application to predict a student's final semester percentage based on continuous academic and behavioral assessments. Developed for Bhadrak Autonomous College, Odisha, it employs advanced ensemble learning techniques to provide educators with actionable insights categorized into distinct performance bands (Excellent, Good, Average, Poor).

## Problem Statement

Early identification of at-risk students remains a challenge in higher education. This project aims to bridge the gap between continuous assessment data and final outcomes by building a responsive, predictive Machine Learning pipeline capable of proactive student performance evaluation.

## Objectives

- Perform comprehensive Exploratory Data Analysis (EDA) on 20,000 student records.
- Preprocess data systematically handling missing values, encoding categorical inputs, and scaling numerical parameters.
- Train, evaluate, and compare multiple regression algorithms (Linear Regression, Random Forest Regressor, Gradient Boosting Regressor).
- Deploy the optimal model through an interactive, modern Streamlit web interface.

## Methodology

1. **Data Collection & Preparation**: Extracted from `student_performance_dataset.xlsx`. Filtered redundant fields (Name, Roll Number).
2. **Exploratory Data Analysis (EDA)**: Analyzed feature importance, inter-variable correlations, and target distribution using Matplotlib and Seaborn.
3. **Data Preprocessing**: Defined an `sklearn.compose.ColumnTransformer` leveraging `StandardScaler` for numeric variables and `OneHotEncoder` for strings/categoricals.
4. **Model Engineering**: Configured `sklearn.pipeline.Pipeline` with hyper-parameterized models.
5. **Evaluation Metrics**: Tested models simultaneously on R² Score, Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
6. **Deployment**: Exported the best performing pipeline using `joblib` and served via `Streamlit` incorporating extensive custom CSS styling for a production-grade UI.

## Results

The regression models were successfully trained. The **Gradient Boosting Regressor / Random Forest** demonstrated the highest R² score, capturing the underlying variance efficiently. Full visualizations including Actual vs Predicted outputs and Feature Importances are preserved within the `visualizations/` namespace and integrated into the Streamlit Data Insights console.

## Conclusion

The developed system demonstrates high reliability in estimating student success, allowing college administration to systematically trigger interventions for students requiring academic support. The modular codebase respects software engineering best practices, making it highly robust.

## Future Scope

- Integration with the University ERP system for real-time telemetry.
- Expand classification boundaries to incorporate multi-label clustering.
- Deep Learning (Neural Networks) implementation for unstructured text review processing.
- Multi-tenant dashboard access with role-based authentication.

---

**Maintained by:** Senior Developer & Data Scientist.
