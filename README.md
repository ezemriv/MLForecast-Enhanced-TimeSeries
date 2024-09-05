<img src="https://img.shields.io/badge/Python-white?logo=Python" style="height: 25px; width: auto;">  <img src="https://img.shields.io/badge/pandas-white?logo=pandas&logoColor=250458" style="height: 25px; width: auto;">  <img src="https://img.shields.io/badge/NumPy-white?logo=numpy&logoColor=013243" style="height: 25px; width: auto;">  <img src="https://img.shields.io/badge/Scikit_learn-white?logo=scikitlearn&logoColor=F7931E" style="height: 25px; width: auto;">

# Multivariate Time Series Sales Forecasting with MLForecast

## Description

This project delves into forecasting sales from multiple multivariate time series leveraging the MLForecast library. MLForecast streamlines time series forecasting by simplifying data handling, automating target-derived feature engineering, and offering multi-model support for a comprehensive analysis.

## Key Features of MLForecast

- **Multi-model support:** Train and compare diverse machine learning models (e.g., LightGBM, XGBoost) for optimal results.
- **Automated feature engineering:** Automatically generates time-based features, saving manual effort.
- **Scalability:** Handles massive datasets and large-scale forecasting problems efficiently.
- **Flexibility:** Accommodates various ML models, including tree-based models and neural networks.
- **Integration with popular ML libraries:** Works seamlessly with scikit-learn, LightGBM, etc.

Example of MLForecast object instantiation including several models, feature engineering, and target transformation:

```python
fcst = MLForecast(
    models=models,
    freq='M',
    lags=[1,2],
    lag_transforms={
        1: [
            (rolling_mean, 3),
            (rolling_std, 3),
            ExpandingMean(),
            ExponentiallyWeightedMean(alpha=0.5),
            ExpandingStd(),
            diff_over_previous, (diff_over_previous, 2)
        ],
        2: [
            (rolling_mean, 3),
            (rolling_std, 3),
            ExpandingMean(),
            ExponentiallyWeightedMean(alpha=0.5),
            ExpandingStd(),
            diff_over_previous, (diff_over_previous, 2)
        ],
    },
    target_transforms=[GlobalSklearnTransformer(sk_log1p), Differences([1])],
    # Note that MLForecast takes care of transforming predictions back to the original scale.
)
```

## Project Structure

### 00 - Exploratory Data Analysis (EDA) and Imputation
- Performs EDA to understand the time series data.
- Employs KNN imputation with custom date features for missing value treatment.

### 01 - MLForecast_Evaluator Class & Advanced Feature Engineering
- Implements a custom class to evaluate models created with MLForecast.
- Introduces advanced feature engineering techniques.
- Tests XGBoost and CatBoost models.
- Performs hyperparameter tuning using Optuna:
  - **Benefits of Optuna:** Efficiently explores the hyperparameter space, automates the optimization process, and often finds better model configurations than manual tuning.

### 02 - Recursive Feature Elimination
- Applies recursive feature elimination to identify the most impactful features.
- Achieves comparable results with a reduced set of features (30 vs. 150+).

For further insights and advanced usage, refer to the well-documented MLForecast resources at https://nixtlaverse.nixtla.io/mlforecast
