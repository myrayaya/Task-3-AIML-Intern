# IMPORTING LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# LOAD THE DATASET
df = pd.read_csv("dataset/Housing.csv")

print(df.head())
print(df.info())

# ENCODE YES/NO CATEGORICALS
for col in ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# ENCODE FURNISHINGSTATUS
df['furnishingstatus'] = df['furnishingstatus'].map({'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2})

# SIMPLE LINEAR REGRESSION
x_simple = df[['area']]
y = df['price']

x_train_s, x_test_s, y_train_s, y_test_s = train_test_split(x_simple, y, test_size=0.2, random_state=42)
model_simple = LinearRegression()
model_simple.fit(x_train_s, y_train_s)
y_pred_simple = model_simple.predict(x_test_s)

# MULTIPLE LINEAR REGRESSION
features = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement',
            'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']
x = df[features]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# EVALUATION
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))
print("Coefficients:", list(zip(features, model.coef_)))

# SIMPLE REGRESSION GRAPH
plt.scatter(x_simple, y, color='blue', alpha=0.5, label='Actual')
plt.plot(x_simple, model_simple.predict(x_simple), color='red', linewidth=2, label='Regression Line')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Simple Linear Regression: Area vs Price')
plt.legend()
plt.show()

# MULTIPLE REGRESSION GRAPH: Predicted vs Actual Prices
plt.scatter(y_test, y_pred, color='green', alpha=0.6)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Multiple Regression: Actual vs Predicted Price')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label='Ideal fit')
plt.legend()
plt.show()
