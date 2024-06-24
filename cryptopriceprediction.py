### Importing libraries
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

"""### Data preprocessing"""

df = pd.read_csv('crypto-markets.csv')
df.head()

df.info()
df.describe()

df.isna().sum()

#Detect and handle outliers
z_scores = stats.zscore(df.select_dtypes(include=['float64', 'int64']))
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
df = df[filtered_entries]

#Scale numerical features
scaler = StandardScaler()
df[df.select_dtypes(include=['float64', 'int64']).columns] = scaler.fit_transform(df.select_dtypes(include=['float64', 'int64']))

df

"""### visualzation"""

fig = go.Figure(data=[go.Scatter(y=df['close'])])
fig.update_layout(title='Crypto Close price.', title_font_size=15, yaxis_title='Price')
fig.show()

fig = go.Figure(data=[go.Scatter(x=df['name'], y=df['close'], name='Closing Price')])
fig.update_layout(title='Closing Price Trend', xaxis_title='name', yaxis_title='Closing Price')
fig.show()

fig = go.Figure(data=[go.Scatter(x=df['date'], y=df['close'], name='Closing Price')])
fig.update_layout(title='Closing Price Trend Over Time', xaxis_title='Date', yaxis_title='Closing Price')
fig.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['close'], bins=50, kde=True)
plt.title('Distribution of Closing Prices')
plt.xlabel('Closing Price')
plt.ylabel('Frequency')
plt.show()

df_positive_ranknow = df[df['ranknow'] > 0]

ranknow_name = df_positive_ranknow.groupby('name')['ranknow'].sum()
sample_data = ranknow_name.sample(20)

colors = sns.color_palette('magma', len(sample_data))
fig = go.Figure(data=[go.Pie(labels=sample_data.index, values=sample_data.values, hole=0.5)])
fig.update_layout(title='Proportion of ranknow by Name')
fig.show()

"""### Feature selection

Correlation Coefficient
"""

numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
corr_matrix = df[numerical_features].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

"""Lasso Regression"""

numerical_features = df.select_dtypes(include=['float64', 'int64']).columns

X = df[numerical_features].drop(columns=['close'])
y = df['close']

lasso = Lasso(alpha=0.01)
lasso.fit(X, y)

model = SelectFromModel(lasso, prefit=True)
selected_features = X.columns[model.get_support()]
print('Selected features by Lasso:', selected_features)

"""### Models"""

features = ['open', 'high', 'low', 'volume', 'market', 'spread']

X = df[features]
y = df['close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""### Linear Regression"""

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

LRmae = mean_absolute_error(y_test, y_pred)
LRmse = mean_squared_error(y_test, y_pred)
LRrmse = np.sqrt(LRmse)
LRr2 = r2_score(y_test, y_pred)

print('Mean Absolute Error:', LRmae)
print('Mean Squared Error:', LRmse)
print('Root Mean Squared Error:', LRrmse)
print('R-squared:', LRr2)

results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results_df = results_df.sort_index()

results_df = results_df.merge(df[['date']], left_index=True, right_index=True)
results_df = results_df.sort_values('date')

sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))
sns.lineplot(data=results_df, x=df['date'], y=results_df['Predicted'], label='Predicted Price', color='blue')

plt.xlabel('Date')
plt.ylabel('Predicted')
plt.title('Predicted Price Over Time')
plt.show()

results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results_df = results_df.sort_index()

results_df = results_df.merge(df[['date']], left_index=True, right_index=True)
results_df = results_df.sort_values('date')

plt.figure(figsize=(14, 8))
plt.plot(results_df['date'], results_df['Actual'], label='Actual Prices', color='black', linestyle='-')
plt.plot(results_df['date'], results_df['Predicted'], label='Linear Regression Predicted Prices', linestyle='--', color='blue')

plt.title('Actual vs Predicted Prices (Linear Regression) - Time Series')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"""### Decision Tree"""

tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

y_pred = tree_model.predict(X_test)

DTmae = mean_absolute_error(y_test, y_pred)
DTmse = mean_squared_error(y_test, y_pred)
DTrmse = np.sqrt(DTmse)
DTr2 = r2_score(y_test, y_pred)

print('Mean Absolute Error:', DTmae)
print('Mean Squared Error:', DTmse)
print('Root Mean Squared Error:', DTrmse)
print('R-squared:', DTr2)

results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results_df = results_df.sort_index()

results_df = results_df.merge(df[['date']], left_index=True, right_index=True)
results_df = results_df.sort_values('date')

plt.figure(figsize=(14, 8))

plt.plot(results_df['date'], results_df['Actual'], label='Actual Prices', color='black', linestyle='-')
plt.plot(results_df['date'], results_df['Predicted'], label='Decision Tree Predicted Prices', linestyle='--', color='blue')

plt.title('Actual vs Predicted Prices (Decision Tree) - Time Series')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"""### Random Forest"""

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

RFmae = mean_absolute_error(y_test, y_pred)
RFmse = mean_squared_error(y_test, y_pred)
RFrmse = np.sqrt(RFmse)
RFr2 = r2_score(y_test, y_pred)

print('Mean Absolute Error:', RFmae)
print('Mean Squared Error:', RFmse)
print('Root Mean Squared Error:', RFrmse)
print('R-squared:', RFr2)

sample_size = 500
sample_indices = np.random.choice(len(y_test), sample_size, replace=False)
y_test_sample = y_test.iloc[sample_indices]
y_pred_sample = y_pred[sample_indices]

plt.figure(figsize=(14, 8))

plt.plot(y_test_sample.index, y_test_sample, label='Actual Prices', color='black', linestyle='-')
plt.plot(y_test_sample.index, y_pred_sample, label='Predicted Prices', linestyle='--', color='blue')

plt.title('Actual vs Predicted Prices')
plt.xlabel('Index')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results_df = results_df.sort_index()

results_df = results_df.merge(df[['date']], left_index=True, right_index=True)
results_df = results_df.sort_values('date')

plt.figure(figsize=(14, 8))
plt.plot(results_df['date'], results_df['Actual'], label='Actual Prices', color='black', linestyle='-')
plt.plot(results_df['date'], results_df['Predicted'], label='Random Forest Predicted Prices', linestyle='--', color='blue')

plt.title('Actual vs Predicted Prices (Random Forest) - Time Series')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"""### Bayesian Ridge Regression"""

bayesian_model = BayesianRidge()
bayesian_model.fit(X_train, y_train)
y_pred = bayesian_model.predict(X_test)

BRmae = mean_absolute_error(y_test, y_pred)
BRmse = mean_squared_error(y_test, y_pred)
BRrmse = np.sqrt(BRmse)
BRr2 = r2_score(y_test, y_pred)

print('Mean Absolute Error:', BRmae)
print('Mean Squared Error:', BRmse)
print('Root Mean Squared Error:', BRrmse)
print('R-squared:', BRr2)

results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results_df = results_df.sort_index()

results_df = results_df.merge(df[['date']], left_index=True, right_index=True)
results_df = results_df.sort_values('date')

plt.figure(figsize=(14, 8))

plt.plot(results_df['date'], results_df['Actual'], label='Actual Prices', color='black', linestyle='-')
plt.plot(results_df['date'], results_df['Predicted'], label='Bayesian Ridge Regression Predicted Prices', linestyle='--', color='blue')

plt.title('Actual vs Predicted Prices (Bayesian Ridge Regression) - Time Series')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"""SVR"""

svr_model = SVR(kernel='rbf')
svr_model.fit(X_train, y_train)
y_pred = svr_model.predict(X_test)

SVRmae = mean_absolute_error(y_test, y_pred)
SVRmse = mean_squared_error(y_test, y_pred)
SVRrmse = np.sqrt(SVRmse)
SVRr2 = r2_score(y_test, y_pred)

print('Mean Absolute Error:', SVRmae)
print('Mean Squared Error:', SVRmse)
print('Root Mean Squared Error:', SVRrmse)
print('R-squared:', SVRr2)

"""### Gradient Boosting"""

gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)
y_pred = gb_model.predict(X_test)

GBmae = mean_absolute_error(y_test, y_pred)
GBmse = mean_squared_error(y_test, y_pred)
GBrmse = np.sqrt(GBmse)
GBr2 = r2_score(y_test, y_pred)

print('Mean Absolute Error:', GBmae)
print('Mean Squared Error:', GBmse)
print('Root Mean Squared Error:', GBrmse)
print('R-squared:', GBr2)

results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results_df = results_df.sort_index()

results_df = results_df.merge(df[['date']], left_index=True, right_index=True)
results_df = results_df.sort_values('date')

plt.figure(figsize=(14, 8))

plt.plot(results_df['date'], results_df['Actual'], label='Actual Prices', color='black', linestyle='-')
plt.plot(results_df['date'], results_df['Predicted'], label='Gradient Boosting Regressor Predicted Prices', linestyle='--', color='blue')

plt.title('Actual vs Predicted Prices (Gradient Boosting Regressor) - Time Series')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"""### XGBoost Regressor"""

xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

XGmae = mean_absolute_error(y_test, y_pred)
XGmse = mean_squared_error(y_test, y_pred)
XGrmse = np.sqrt(XGmse)
XGr2 = r2_score(y_test, y_pred)

print('Mean Absolute Error:', XGmae)
print('Mean Squared Error:', XGmse)
print('Root Mean Squared Error:', XGrmse)
print('R-squared:', XGr2)

results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results_df = results_df.sort_index()

results_df = results_df.merge(df[['date']], left_index=True, right_index=True)
results_df = results_df.sort_values('date')

plt.figure(figsize=(14, 8))
plt.plot(results_df['date'], results_df['Actual'], label='Actual Prices', color='black', linestyle='-')
plt.plot(results_df['date'], results_df['Predicted'], label='XGBoost Regressor Predicted Prices', linestyle='--', color='blue')

plt.title('Actual vs Predicted Prices (XGBoost Regressor) - Time Series')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"""### Results"""

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