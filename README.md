# Car Price Project

This project focuses on EDA and predicting the price of used cars based on various features using machine learning techniques. The goal is to build a model that can accurately predict car prices based on factors such as make, model, year, mileage, and more.

## Table of Contents
- [Project Overview](#project-overview)
- [Data](#data)
- [Data Cleaning](#data-cleaning)
- [Machine Learning Models](#machine-learning-models)
- [Results](#results)
- [Installation](#installation)



## Project Overview

In this project, a dataset of used cars is used to build regression models that predict the price of cars based on a set of features. Various models were tested, including:
- **Linear Regression**
- **Lasso Regression**
- **Random Forest Regressor**
- **XGBoost Regressor**

Hyperparameter tuning was also performed using RandomizedSearchCV to optimize the models for better performance.

## Data

The dataset used in this project is called `used_cars.csv`, which contains several features about used cars. These features include:
- `make`: Car brand 
- `model`: Car model 
- `year`: Year of manufacture
- `mileage`: Distance the car has traveled in kilometers
- `price`: The target variable (price of the car)

The dataset can be found in https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset/data and is used to train and test the models.

## Data Cleaning

- Handling Missing Values
- Checking for duplicates
- Feature Engineering

## Machine Learning Models

In this project, we tested several machine learning models to predict car prices. The models used are:

1. **Linear Regression**: A simple linear model used for predicting a target variable (price) based on linear relationships between the features.
2. **Lasso Regression**: A type of linear regression that uses L1 regularization to help reduce overfitting and select important features.
3. **Random Forest Regressor**: A tree-based ensemble method that can model complex non-linear relationships.
4. **XGBoost Regressor**: A gradient boosting method known for its high performance in many regression tasks.

**Hyperparameter Tuning** was performed using **RandomizedSearchCV** to find the best model parameters.

## Results

The models' performance was evaluated using the following metrics:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**

Results were compared between the different models to choose the best one for predicting car prices. The XGBoost Regressor provided the best performance after hyperparameter tuning.
Best Parameters:  {'xgb__subsample': 0.7, 'xgb__reg_lambda': 1.5, 'xgb__reg_alpha': 0, 'xgb__n_estimators': 300, 'xgb__min_child_weight': 1, 'xgb__max_depth': 6, 'xgb__learning_rate': 0.1, 'xgb__gamma': 0.3, 'xgb__colsample_bytree': 0.8}
MAE: 5435.93
MSE: 61767264.00
RMSE: 7859.22
R² Score: 0.86

### Top 20 Car Brands

![top20_car_brands](https://github.com/user-attachments/assets/0b9f1b6f-3289-4782-9985-e034736b3cd0)

### Top 10 Interior Colors

![top10_interior_colors](https://github.com/user-attachments/assets/4720b5b8-63ed-4610-8e90-add9fd78f53b)

### Top 10 Exterior Colors

![top10_exterior_colors](https://github.com/user-attachments/assets/094c1cf6-9844-4e63-aa6a-58ba6b9baf3b)

### Scatter plot of Car Price vs Mileage

![car_price_vs_mileage](https://github.com/user-attachments/assets/71fae68f-d140-4507-bf40-1e2e60ea6640)

### Distribution of Price by Accident

![price_distribution_by_accident](https://github.com/user-attachments/assets/bcbe3a64-56cc-43dd-9ffe-383078d003fa)

### Distribution of Mileage by Model Year

![distribution_of_mileage_by_model_year](https://github.com/user-attachments/assets/a4de1ca8-9d6c-4fed-8f1a-1d434fbacc2d)

### Correlation Matrix

![correlation_matrix](https://github.com/user-attachments/assets/f15f409c-574a-4b67-8ca9-deb82a92020b)

These visualizations provide insights of the dataset.

## Installation

To run this project, make sure you have Python installed, and then install the required dependencies. If you're using a virtual environment, activate it before proceeding.

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/MichailidisData/Car-price-project.git
   cd Car-price-project
