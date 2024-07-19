Project Title: Multivariate Time Series Sales Forecasting with MLForecast
sdf
Description

This project delves into forecasting sales from multiple multivariate time series leveraging the MLForecast library. MLForecast streamlines time series forecasting by simplifying data handling, automating feature engineering, and offering multi-model support for a comprehensive analysis.

Key Features of MLForecast

Multi-model support: Train and compare diverse machine learning models (e.g., LightGBM, XGBoost) for optimal results.
Automated feature engineering: Automatically generates time-based features, saving manual effort.
Scalability: Handles massive datasets and large-scale forecasting problems efficiently.
Flexibility: Accommodates various ML models, including tree-based models and neural networks.
Integration with popular ML libraries: Works seamlessly with scikit-learn, LightGBM, etc.
Project Structure

00 - Exploratory Data Analysis (EDA) and Imputation:
Performs EDA to understand the time series data.
Employs KNN imputation with custom date features for missing value treatment.
01 - MLForecast_Evaluator Class & Advanced Feature Engineering:
Implements a custom class to evaluate models created with MLForecast.
Introduces advanced feature engineering techniques.
Tests XGBoost and CatBoost models.
Performs hyperparameter tuning using Optuna:
Benefits of Optuna: Efficiently explores the hyperparameter space, automates the optimization process, and often finds better model configurations than manual tuning.
02 - Recursive Feature Elimination:
Applies recursive feature elimination to identify the most impactful features.
Achieves comparable results with a reduced set of features (30 vs. 150+).
Installation

Clone this repository.
Install required libraries using pip install -r requirements.txt
Usage

Explore the notebooks sequentially (00, 01, 02).
Adapt the code to your specific dataset and forecasting goals.
Experiment with different models and features, leveraging MLForecast's flexibility.
Utilize the MLForecast_Evaluator class and Optuna for comprehensive model assessment and tuning.
Consider feature selection techniques like recursive feature elimination for optimization.
Let me know if you have any other details you'd like to include or adjustments you'd like to make!