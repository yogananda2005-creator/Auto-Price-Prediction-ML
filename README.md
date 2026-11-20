# 🚗 Auto Price Prediction Using Machine Learning

## Project Overview

Auto Price Prediction is a Machine Learning regression project developed to predict automobile prices based on various vehicle specifications, engine characteristics, dimensions, fuel efficiency and other technical attributes.

The project uses the historical **1985 Automobile Imports Dataset** to analyze the relationship between vehicle features and their prices and develop predictive regression models.

The primary objective is to build a reliable and reproducible machine learning pipeline that can estimate automobile prices, compare multiple regression algorithms, identify important price-driving factors and provide useful insights for automobile dealers, insurers and importers.

This project is developed as part of the **PRCP-1017 Auto Price Prediction** assignment.

## 🎯 Project Objectives

- Analyze automobile specifications and their relationship with vehicle prices.
- Perform data validation, cleaning and preprocessing.
- Handle missing values and categorical variables.
- Perform Exploratory Data Analysis (EDA).
- Apply feature engineering to create meaningful vehicle attributes.
- Train and compare multiple Machine Learning regression algorithms.
- Perform 5-fold cross-validation.
- Apply hyperparameter tuning to improve model performance.
- Evaluate the final model using appropriate regression metrics.
- Implement Explainable AI using SHAP and permutation feature importance.
- Build and save a reusable end-to-end Machine Learning pipeline.

## 📊 Dataset Description

The project uses the **1985 Auto Imports Database**, associated with Jeffrey C. Schlimmer (1987).

Dataset details:

| Attribute | Description |
|---|---|
| Dataset | 1985 Automobile Imports |
| Available records | 201 |
| Total attributes | 26 |
| Numerical attributes | 16 |
| Categorical attributes | 10 |
| Target variable | `price` |
| Problem type | Supervised Machine Learning |
| ML task | Regression |

The dataset contains information about:

- Automobile manufacturers and brands.
- Fuel type and aspiration.
- Body style and drive wheels.
- Engine type and number of cylinders.
- Engine size and horsepower.
- Vehicle dimensions and weight.
- Fuel efficiency.
- Insurance risk rating and normalized losses.
- Automobile prices.

**Target Variable:** `price`

The model uses the available vehicle specifications to predict the automobile price.

## 🛠️ Technologies and Tools Used

**Programming Language**
- Python

**Data Analysis and Manipulation**
- Pandas
- NumPy

**Data Visualization**
- Matplotlib
- Seaborn

**Machine Learning**
- Scikit-learn
- XGBoost (optional)

**Model Explainability**
- SHAP
- Permutation Feature Importance

**Model Development**
- Scikit-learn Pipelines
- ColumnTransformer
- FunctionTransformer
- Cross-Validation
- Grid/Randomized Hyperparameter Search

**Model Persistence**
- Joblib

**Development Environment**
- Jupyter Notebook / Google Colab

## 🔍 Project Workflow

### 1. Data Loading and Validation

- Imported the automobile dataset.
- Assigned column names using the dataset documentation.
- Validated dataset dimensions and column structure.
- Checked numerical and categorical data types.
- Verified the target variable.
- Recorded a SHA-256 fingerprint to verify the original dataset remains unchanged.

### 2. Data Cleaning and Preprocessing

- Identified missing-value placeholders represented by `?`.
- Converted missing-value markers into NaN.
- Converted numerical columns into appropriate numerical data types.
- Analyzed duplicate records.
- Checked invalid values and documented ranges.
- Preserved genuine observations instead of unnecessarily deleting records.
- Prepared separate preprocessing pipelines for numerical and categorical features.

### 3. Exploratory Data Analysis (EDA)

Performed detailed exploratory analysis to understand the dataset.

Key activities:

- Numerical feature distribution analysis.
- Categorical feature distribution analysis.
- Automobile price distribution analysis.
- Correlation analysis.
- Spearman and Pearson correlation comparisons.
- Multicollinearity analysis.
- Variance Inflation Factor (VIF).
- Outlier detection using the IQR method.
- Relationship analysis between vehicle specifications and automobile prices.

The analysis identified important relationships between automobile prices, engine size, curb weight, horsepower, fuel efficiency and manufacturer.

### 4. Feature Engineering

Created additional features to improve the predictive capability of the models.

| Engineered Feature | Description |
|---|---|
| cylinders | Converts cylinder categories into numerical values |
| doors | Converts door categories into numerical values |
| power-to-weight | Horsepower divided by curb weight |
| engine-size-per-cylinder | Engine size divided by cylinder count |
| body-volume-k | Length × Width × Height / 1000 |
| avg-mpg | Average of city and highway mileage |
| mpg-gap | Difference between highway and city mileage |
| bore-stroke-ratio | Bore divided by stroke |
| premium-make | Identifies selected premium automobile manufacturers |

These features help represent vehicle performance, dimensions, engine characteristics and brand positioning.

### 5. Data Preprocessing

Implemented a structured preprocessing pipeline.

- Numerical feature imputation.
- Categorical feature imputation.
- Missing-value indicators.
- Numerical feature scaling.
- One-hot encoding for categorical variables.
- Handling previously unseen categories.
- Training-only fitting of preprocessing transformations to prevent data leakage.

### 6. Train-Test Split

Separated the dataset into training and testing subsets.

- Training data: 80%
- Testing data: 20%
- Random state: 42
- Cross-validation: 5-fold

The test dataset is kept separate during model selection to provide an independent final evaluation.

### 7. Machine Learning Model Development

Implemented and compared multiple regression algorithms.

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. ElasticNet Regression
5. K-Nearest Neighbors Regression
6. Decision Tree Regression
7. Random Forest Regression
8. Extra Trees Regression
9. Gradient Boosting Regression
10. HistGradientBoosting Regression
11. XGBoost Regression (optional, depending on library availability)

These algorithms help evaluate linear relationships, regularized regression methods, distance-based prediction and tree-based ensemble learning.

### 8. Model Evaluation and Cross-Validation

Evaluated model performance using multiple regression metrics.

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

Implemented 5-fold cross-validation to compare model consistency and generalization performance.

MAE is used as the primary metric for model selection, while RMSE and R² provide additional performance insights.

### 9. Hyperparameter Tuning

Applied hyperparameter optimization to promising models.

The objective is to:

- Identify suitable model parameters.
- Reduce prediction errors.
- Improve generalization.
- Control overfitting.
- Select the best-performing model using cross-validation results.

### 10. Final Model Evaluation

Evaluated the selected model using the untouched test dataset.

Generated:

- Actual versus predicted automobile prices.
- Prediction errors.
- Absolute prediction errors.
- Final MAE.
- Final MSE.
- Final RMSE.
- Final R² score.
- Median Absolute Error.
- Mean Absolute Percentage Error (MAPE).

### 11. Explainable AI (XAI)

Implemented model interpretability techniques to understand the factors influencing automobile price predictions.

**Permutation Feature Importance**
- Identifies influential input features.
- Measures the effect of feature permutation on model performance.
- Helps understand the contribution of vehicle specifications.

**SHAP (SHapley Additive exPlanations)**
- Explains individual model predictions.
- Identifies positive and negative feature contributions.
- Provides global feature importance insights.
- Uses TreeExplainer for supported tree-based models and a model-agnostic explainer otherwise.

SHAP execution is optional and depends on library availability and model compatibility.

### 12. Model Saving and Production Pipeline

Saved the complete fitted Machine Learning pipeline using Joblib.

The saved pipeline includes:

- Feature engineering.
- Data preprocessing.
- Trained regression model.

The project also verifies that the saved pipeline can be reloaded and reproduces the original test predictions.

## 📈 Expected Project Deliverables

The notebook generates the following outputs after successful execution:

- `auto_price_model.joblib`
- `model_comparison.csv`
- `combined_model_comparison.csv`
- `tuned_model_comparison.csv`
- `predictions.csv`
- `feature_importance.csv`
- `shap_importance.csv` (optional)
- `final_metrics.json`
- `challenges_report.csv`

## 💡 Business Applications

This project demonstrates how Machine Learning can support:

- Preliminary automobile price estimation.
- Used-car valuation analysis.
- Vehicle listing price comparisons.
- Automobile inventory management.
- Identification of pricing inconsistencies.
- Understanding vehicle specifications that influence price.
- Data-driven pricing decisions.

The model is intended for historical-data-based analysis and preliminary estimates, not as a replacement for current-market vehicle valuation.

## ⚠️ Challenges Addressed

- Missing values represented by special characters.
- Mixed numerical and categorical features.
- Different feature scales.
- Multicollinearity between vehicle specifications.
- Outliers in automobile prices.
- Rare categorical values.
- Potential model overfitting.
- Model interpretability.
- Limited historical dataset size.

## 🚀 Future Improvements

- Integrate larger and more recent automobile datasets.
- Develop an interactive automobile price prediction application.
- Build a FastAPI-based prediction service.
- Deploy the model using a cloud platform.
- Implement real-time price prediction.
- Improve model generalization with additional vehicle data.
- Monitor prediction performance using new market information.

## 📌 Conclusion

The Auto Price Prediction project demonstrates an end-to-end Machine Learning regression workflow, beginning with raw automobile data and progressing through data validation, exploratory analysis, feature engineering, preprocessing, model development, cross-validation, hyperparameter tuning, model evaluation and explainable AI.

By comparing multiple regression algorithms and selecting a model based on cross-validation performance, the project establishes a structured approach to automobile price estimation.

It also demonstrates practical applications of Python, Data Science, Machine Learning, feature engineering and model interpretability.

**Project:** PRCP-1017 – Auto Price Prediction  
**Domain:** Machine Learning | Data Science | Regression  
**Programming Language:** Python  
**Repository:** Auto-Price-Prediction-ML
