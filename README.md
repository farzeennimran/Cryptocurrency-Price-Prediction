# Cryptocurrency Price Prediction

## Introduction

This repository contains a project focused on predicting cryptocurrency prices using various machine learning models. The goal is to analyze historical cryptocurrency data and build predictive models to estimate future prices. The dataset used in this project is `crypto-markets.csv`, taken from kaggle, which contains various features such as open, high, low, close prices, volume, market cap, and rank. 

The project includes the following steps:
1. Data Preprocessing
2. Data Visualization
3. Feature Selection
4. Model Training and Evaluation
5. Results Visualization

## Code Explanation

### Importing Libraries

```python
import numpy as np
import pandas as pd
import seaborn as sb
import seaborn as sns
from scipy import stats
import plotly.express as px
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import plotly.graph_objects as go
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

### Data Preprocessing

1. **Loading Data:**
   ```python
   df = pd.read_csv('crypto-markets.csv')
   df.head()
   ```

2. **Basic Data Analysis:**
   ```python
   df.info()
   df.describe()
   df.isna().sum()
   ```

3. **Handling Outliers:**
   ```python
   z_scores = stats.zscore(df.select_dtypes(include=['float64', 'int64']))
   abs_z_scores = np.abs(z_scores)
   filtered_entries = (abs_z_scores < 3).all(axis=1)
   df = df[filtered_entries]
   ```

4. **Scaling Numerical Features:**
   ```python
   scaler = StandardScaler()
   df[df.select_dtypes(include=['float64', 'int64']).columns] = scaler.fit_transform(df.select_dtypes(include=['float64', 'int64']))
   ```

### Data Visualization

1. **Close Price Scatter Plot:**
   ```python
   fig = go.Figure(data=[go.Scatter(y=df['close'])])
   fig.update_layout(title='Crypto Close Price', yaxis_title='Price')
   fig.show()
   ```

2. **Closing Price Trend:**
   ```python
   fig = go.Figure(data=[go.Scatter(x=df['name'], y=df['close'], name='Closing Price')])
   fig.update_layout(title='Closing Price Trend', xaxis_title='Name', yaxis_title='Closing Price')
   fig.show()
   ```

3. **Closing Price Over Time:**
   ```python
   fig = go.Figure(data=[go.Scatter(x=df['date'], y=df['close'], name='Closing Price')])
   fig.update_layout(title='Closing Price Trend Over Time', xaxis_title='Date', yaxis_title='Closing Price')
   fig.show()
   ```

4. **Distribution of Closing Prices:**
   ```python
   plt.figure(figsize=(10, 6))
   sns.histplot(df['close'], bins=50, kde=True)
   plt.title('Distribution of Closing Prices')
   plt.xlabel('Closing Price')
   plt.ylabel('Frequency')
   plt.show()
   ```

5. **Proportion of Rank Now by Name:**
   ```python
   df_positive_ranknow = df[df['ranknow'] > 0]
   ranknow_name = df_positive_ranknow.groupby('name')['ranknow'].sum()
   sample_data = ranknow_name.sample(20)
   colors = sns.color_palette('magma', len(sample_data))
   fig = go.Figure(data=[go.Pie(labels=sample_data.index, values=sample_data.values, hole=0.5)])
   fig.update_layout(title='Proportion of Rank Now by Name')
   fig.show()
   ```

### Feature Selection

1. **Correlation Matrix:**
   ```python
   numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
   corr_matrix = df[numerical_features].corr()
   plt.figure(figsize=(12, 8))
   sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
   plt.title('Correlation Matrix')
   plt.show()
   ```

2. **Lasso Regression:**
   ```python
   numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
   X = df[numerical_features].drop(columns=['close'])
   y = df['close']
   lasso = Lasso(alpha=0.01)
   lasso.fit(X, y)
   model = SelectFromModel(lasso, prefit=True)
   selected_features = X.columns[model.get_support()]
   print('Selected features by Lasso:', selected_features)
   ```

### Models

The following machine learning models were used to predict cryptocurrency prices:

1. **Linear Regression:**
   ```python
   model = LinearRegression()
   model.fit(X_train, y_train)
   y_pred = model.predict(X_test)
   ```

2. **Decision Tree:**
   ```python
   tree_model = DecisionTreeRegressor(random_state=42)
   tree_model.fit(X_train, y_train)
   y_pred = tree_model.predict(X_test)
   ```

3. **Random Forest:**
   ```python
   rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
   rf_model.fit(X_train, y_train)
   y_pred = rf_model.predict(X_test)
   ```

4. **Bayesian Ridge Regression:**
   ```python
   bayesian_model = BayesianRidge()
   bayesian_model.fit(X_train, y_train)
   y_pred = bayesian_model.predict(X_test)
   ```

5. **Support Vector Regression (SVR):**
   ```python
   svr_model = SVR(kernel='rbf')
   svr_model.fit(X_train, y_train)
   y_pred = svr_model.predict(X_test)
   ```

6. **Gradient Boosting:**
   ```python
   gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
   gb_model.fit(X_train, y_train)
   y_pred = gb_model.predict(X_test)
   ```

7. **XGBoost Regressor:**
   ```python
   xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
   xgb_model.fit(X_train, y_train)
   y_pred = xgb_model.predict(X_test)
   ```

### Results

The performance of each model was evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2) scores. The results were plotted to compare the models.

```python
metrics = {
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'Bayesian Regression', 'Gradient Boosting', 'XGBoost'],
    'MAE': [LRmae, DTmae, RFmae, BRmae, GBmae, XGmae],
    'MSE': [LRmse, DTmse, RFmse, BRmse, GBmse, XGmse],
    'RMSE': [LRrmse, DTrmse, RFrmse, BRrmse, GBrmse, XGrmse],
    'R-squared': [LRr2, DTr2, RFr2, BRr2, GBr2, XGr2]
}

plt.figure(figsize=(14, 10))

# MAE plot
plt.subplot(2, 2, 1)
plt.plot(metrics['Model'], metrics['MAE'], marker='o', linestyle='-', color='b', label='MAE')
plt.title('Mean Absolute Error (MAE) Comparison')
plt.xlabel('Models')
plt.ylabel('MAE')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()

# MSE plot
plt.subplot(2, 2, 2)
plt.plot(metrics['Model'], metrics['MSE'], marker='o', linestyle='-', color='r', label='MSE')
plt.title('Mean Squared Error (MSE) Comparison')
plt.xlabel('Models')
plt.ylabel('MSE')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()

# RMSE plot
plt.subplot(2, 2, 3)
plt.plot(metrics['Model'], metrics['RMSE'], marker='o', linestyle='-', color='g', label='RMSE')
plt.title('Root Mean Squared Error (RMSE) Comparison')
plt.xlabel('Models')
plt.ylabel('RMSE')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()

# R-squared plot
plt.subplot(2, 2, 4)
plt.plot(metrics['Model'], metrics['R-squared'], marker='o', linestyle='-', color='m', label='R-squared')
plt.title('R-squared Comparison')
plt.xlabel('Models')
plt.ylabel('R-squared')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
```

## Conclusion

This project demonstrates the process of predicting cryptocurrency prices using various machine learning models. By preprocessing the data, visualizing trends, selecting relevant features, and applying multiple regression techniques, I was able to build a predictive models to estimate future prices of cryptocurrency. 
