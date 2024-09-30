import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.preprocessing import OneHotEncoder

try:
    linear_reg = pd.read_csv('/home/cs-ai-25/exp-ml/1.01. Simple linear regression.csv')
except FileNotFoundError:
    print('Simple Linear Regression File not found')
    exit()

print('Simple Linear Regression')

X_single = linear_reg.iloc[:,:-1]
y_single = linear_reg.iloc[:,-1]

X_single_train , X_single_test , y_single_train , y_single_test = train_test_split(X_single,y_single,test_size=0.2,random_state=25)

single_model = LinearRegression()
single_model.fit(X_single_train,y_single_train)

y_pred_single = single_model.predict(X_single_test)

mse_single = mean_squared_error(y_single_test,y_pred_single)
mae_single = mean_absolute_error(y_single_test,y_pred_single)
r2_single = r2_score(y_single_test,y_pred_single)

print(f"MSE: {mse_single:.2f}, MAE: {mae_single:.2f}, R2: {r2_single:.2f}")

sat = int(input('Enter SAT score'))
sat_pred = single_model.predict(np.array([[sat]]))

print(f'Predicted GPA: {sat_pred}')

plt.figure(figsize=(16,6))
plt.subplot(1,3,1)
plt.scatter(X_single_test,y_single_test,color='blue')
plt.plot(X_single_test,y_pred_single)
plt.title('Single Linear Regression')
plt.xlabel('SAT')
plt.ylabel('GPA')



print('Multivariable Regression')
try:
    multi = pd.read_csv('/home/cs-ai-25/exp-ml/data_multi.csv')
except FileNotFoundError:
    print('Multivariable Regression File not Found')
    exit()

x_multi = multi.iloc[:,2:-1]
y_multi = multi.iloc[:,-1]


    
X_multi_train,x_multi_test,y_multi_train,y_multi_test = train_test_split(x_multi,y_multi,test_size=.2,random_state=25)

multi_model = LinearRegression()
multi_model.fit(X_multi_train,y_multi_train)

y_pred_multi = multi_model.predict(x_multi_test)



mse_multi = mean_squared_error(y_multi_test,y_pred_multi)
mae_multi = mean_absolute_error(y_multi_test,y_pred_multi)
r2_multi = r2_score(y_multi_test,y_pred_multi)

print(f"MSE: {mse_multi:.2f}, MAE: {mae_multi:.2f}, R2: {r2_multi:.2f}")

volume = input('Volume of Car: ')
weight = input('Weight of Car: ')


user_input = [volume,weight]

input_ = pd.DataFrame([user_input],columns=['Volume','Weight'])



predicted = multi_model.predict(input_)

print(f'Predicted CO2: {predicted[0]}')

plt.subplot(1,3,2)
plt.scatter(y_multi_test,y_pred_multi,color = 'green')
plt.plot([min(y_multi_test),max(y_multi_test)],[min(y_multi_test),max(y_multi_test)],color = 'red')
plt.title('Multivariable Regression')
plt.xlabel('Actual')
plt.ylabel('Predicted')

print('Polynomial Regression')
try:
    poly = pd.read_csv('/home/cs-ai-25/exp-ml/Ice_cream selling data.csv')
except FileNotFoundError:
    print('Polynomial Regession File not found')
    exit()

X = poly[['Temperature (°C)']].values
y = poly['Ice Cream Sales (units)'].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=25)

poly_features = PolynomialFeatures()
X_poly_train = poly_features.fit_transform(X_train)
X_poly_test = poly_features.fit_transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_poly_train,y_train)

y_pred_poly = poly_model.predict(X_poly_test)

mse_poly = mean_squared_error(y_test, y_pred_poly)
mae_poly = mean_absolute_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print(f"Polynomial Regression (Degree 4): MSE: {mse_poly:.2f}, MAE: {mae_poly:.2f}, R2: {r2_poly:.2f}")

X_grid = np.arange(float(X.min()), float(X.max()), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
X_grid_poly = poly_features.transform(X_grid)
y_grid_pred = poly_model.predict(X_grid_poly)

position = int(input('Enter Temprature (°C): '))

salary_pred = poly_model.predict(poly_features.transform([[position]]))
print('Predicted Sales of Units',salary_pred)

plt.subplot(1, 3, 3)
plt.scatter(X, y, color='blue', label='Actual Temprature')
plt.plot(X_grid, y_grid_pred, color='red', label='Polynomial Regression')
plt.title('Polynomial Regression')
plt.xlabel('Temprature')
plt.ylabel('Sales')
plt.legend()
plt.tight_layout()
plt.show()