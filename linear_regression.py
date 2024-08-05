import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import random


n = int(input("Enter the number dataset: "))
array = np.zeros((n,2))

print("Enter the values:")
for i in range(n):
    for j in range(2):
        array[i, j] = float(input())


df = pd.DataFrame(array, columns=[f'Col_{i}' for i in range(2)])

print("\nGenerated DataFrame:")
print(df)


X = df[['Col_0']].values  
y = df['Col_1'].values    


model = LinearRegression()
model.fit(X, y)


y_pred = model.predict(X)


print(f"\nCoefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

a = float(input("Enter the value"))
b = model.predict([[a]])
print(b[0])

plt.scatter(X, y, color='blue', label='Original Data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Fitted Line')
plt.xlabel('Col_0')
plt.ylabel('Col_1')
plt.title('Linear Regression')
plt.legend()
plt.show()