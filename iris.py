import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

iris = load_iris()
data = iris.data

target = iris.target
names = iris.target_names
feature = iris.feature_names

df = pd.DataFrame(data,columns=feature)
df['species']=target
plt.figure(figsize=(12,10))


for species in range(len(names)):
    subset = df[df['species'] == species ]
    plt.scatter(subset[feature[0]], subset[feature[1]], label=target[species])
plt.xlabel(feature[0])
plt.ylabel(feature[1])
plt.title('Sepal Length vs Sepal Width')
plt.legend()
plt.show()